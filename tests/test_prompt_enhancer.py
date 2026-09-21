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


def test_rewrite_preserves_image_order_and_decodes_only_generated_tokens(monkeypatch):
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
        assert kwargs["enable_thinking"] is True
        return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(tokens, **kwargs):
        assert tokens.tolist() == [4, 5]
        return '<think>reasoning</think>{"rewritten_prompt": "Combine the images."}'

    def load(device):
        enhancer.system_prompt = "Official instructions"
        enhancer.processor = SimpleNamespace(
            apply_chat_template=template, tokenizer=SimpleNamespace(decode=decode),
        )
        enhancer.model = SimpleNamespace(
            generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
        )

    monkeypatch.setattr(enhancer, "load", load)
    assert enhancer.rewrite(images, "Combine", "cpu") == "Combine the images."
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

    def rewrite(images, prompt, device):
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


def test_ui_returns_rewritten_prompt(monkeypatch):
    def rewrite(paths, prompt, progress):
        assert paths == ["reference.png"] and prompt == "Edit"
        return "A precise edit"

    monkeypatch.setattr(app.manager, "rewrite_prompt", rewrite)
    assert app.enhance_prompt("Image to Image", ["reference.png"], "Edit") == "A precise edit"


def test_ui_rejects_text_to_image_rewriting():
    with pytest.raises(app.gr.Error, match="Image to Image"):
        app.enhance_prompt("Text to Image", [], "Create a cat")
