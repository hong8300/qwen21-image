"""Prompt rewriting, validation, and model memory lifecycle."""

from types import SimpleNamespace

import pytest
from PIL import Image

import app
from backend import ModelManager
from prompt_enhancer import PromptEnhancer, parse_rewritten_prompt


@pytest.mark.parametrize("answer", [
    '{"rewritten_prompt": "Make the square blue."}',
    '<think>reasoning with {braces}</think>{"rewritten_prompt": "Make the square blue."}',
    'reasoning</think>```json\n{"rewritten_prompt": "Make the square blue."}\n```',
])
def test_parse_final_answer(answer):
    assert parse_rewritten_prompt(answer) == "Make the square blue."


@pytest.mark.parametrize("answer", [
    '<think>unfinished', 'not JSON', '{}', '[]',
    '{"rewritten_prompt": " "}', '{"rewritten_prompt": 123}',
])
def test_reject_missing_or_invalid_answer(answer):
    with pytest.raises(ValueError):
        parse_rewritten_prompt(answer)


@pytest.mark.parametrize("thorough", [False, True])
def test_rewrite_preserves_image_order_and_decodes_only_generated_tokens(monkeypatch, thorough):
    import torch

    enhancer = PromptEnhancer()
    images = [Image.new("RGBA", (32, 32), color) for color in ("red", "blue")]
    captured = {}

    class Inputs(dict):
        def to(self, device):
            captured["device"] = device
            return self

    def template(messages, **kwargs):
        captured["messages"] = messages
        assert kwargs["enable_thinking"] is thorough
        return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(tokens, **kwargs):
        assert tokens.tolist() == [4, 5]
        return '<think>reasoning</think>{"rewritten_prompt": "Combine the images."}'

    def generate(**kwargs):
        assert kwargs["max_new_tokens"] == (24000 if thorough else 2048)
        return torch.tensor([[1, 2, 3, 4, 5]])

    def load(device):
        enhancer.system_prompt = "Official instructions"
        enhancer.processor = SimpleNamespace(
            apply_chat_template=template, tokenizer=SimpleNamespace(decode=decode),
        )
        enhancer.model = SimpleNamespace(generate=generate)

    monkeypatch.setattr(enhancer, "load", load)
    assert enhancer.rewrite(images, "Combine", "cpu", thorough=thorough) == "Combine the images."
    content = captured["messages"][1]["content"]
    assert [item["image"].getpixel((0, 0)) for item in content[:-1]] == [
        (255, 0, 0), (0, 0, 255),
    ]
    assert content[-1] == {"type": "text", "text": "Combine"}
    assert captured["device"] == "cpu"


@pytest.mark.parametrize("fails", [False, True])
def test_manager_releases_models_before_and_after_rewriting(tmp_path, monkeypatch, fails):
    path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(path)
    manager = ModelManager()
    manager.pipeline = object()
    monkeypatch.setattr(manager, "select_device", lambda: "cpu")

    def rewrite(images, prompt, device, **kwargs):
        assert manager.pipeline is None
        assert len(images) == 1 and prompt == "Edit" and device == "cpu"
        manager.enhancer.model = object()
        manager.enhancer.processor = object()
        if fails:
            raise RuntimeError("inference failed")
        return "A precise edit"

    monkeypatch.setattr(manager.enhancer, "rewrite", rewrite)
    if fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            manager.rewrite_prompt([str(path)], " Edit ")
    else:
        assert manager.rewrite_prompt([str(path)], " Edit ") == "A precise edit"
    assert manager.enhancer.model is None
    assert manager.enhancer.processor is None
    assert manager.pipeline is None


@pytest.mark.parametrize(("paths", "prompt"), [
    ([], "Edit"), (["missing.png"] * 11, "Edit"), (["missing.png"], " "),
])
def test_invalid_rewrite_input_does_not_unload_existing_model(paths, prompt):
    manager = ModelManager()
    pipeline = object()
    manager.pipeline = pipeline
    with pytest.raises(ValueError):
        manager.rewrite_prompt(paths, prompt)
    assert manager.pipeline is pipeline


@pytest.mark.parametrize("thorough", [False, True])
def test_ui_returns_rewritten_prompt(monkeypatch, thorough):
    def rewrite(paths, prompt, progress, **kwargs):
        assert paths == ["reference.png"] and prompt == "Edit"
        assert kwargs["thorough"] is thorough
        return "A precise edit"

    monkeypatch.setattr(app.manager, "rewrite_prompt", rewrite)
    assert app.enhance_prompt(
        "Image to Image", ["reference.png"], "Edit", thorough,
    ) == "A precise edit"


def test_ui_rejects_text_to_image_rewriting():
    with pytest.raises(app.gr.Error, match="Image to Image"):
        app.enhance_prompt("Text to Image", [], "Create a cat")


def test_repeated_original_and_rewritten_prompts_reuse_result(tmp_path, monkeypatch):
    path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "red").save(path)
    manager = ModelManager()
    calls = []
    monkeypatch.setattr(manager, "select_device", lambda: "cpu")

    def rewrite(*args, **kwargs):
        calls.append(args)
        return "A precise edit"

    monkeypatch.setattr(manager.enhancer, "rewrite", rewrite)
    assert manager.rewrite_prompt([str(path)], "Edit") == "A precise edit"
    pipeline = object()
    manager.pipeline = pipeline
    for prompt in ("Edit", "A precise edit"):
        assert manager.rewrite_prompt([str(path)], prompt) == "A precise edit"
        assert manager.rewrite_stats["cache_hit"] is True
        assert manager.pipeline is pipeline
    assert len(calls) == 1
    manager.unload()
    manager.rewrite_prompt([str(path)], "Edit")
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["pixels", "order", "prompt", "mode"])
def test_changed_input_does_not_reuse_stale_result(tmp_path, monkeypatch, change):
    paths = [str(tmp_path / f"{color}.png") for color in ("red", "blue")]
    for path, color in zip(paths, ("red", "blue"), strict=True):
        Image.new("RGB", (32, 32), color).save(path)
    manager = ModelManager()
    calls = []
    monkeypatch.setattr(manager, "select_device", lambda: "cpu")

    def rewrite(*args, **kwargs):
        calls.append(args)
        return f"Edit result {len(calls)}"

    monkeypatch.setattr(manager.enhancer, "rewrite", rewrite)
    manager.rewrite_prompt(paths, "Edit")
    if change == "pixels":
        Image.new("RGB", (32, 32), "green").save(paths[0])
    if change == "order":
        paths.reverse()
    result = manager.rewrite_prompt(
        paths, "Different edit" if change == "prompt" else "Edit", thorough=change == "mode",
    )
    assert result == "Edit result 2"
    assert manager.rewrite_stats["cache_hit"] is False
    assert len(manager.rewrite_cache) == 2
