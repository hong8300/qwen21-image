import json
from types import SimpleNamespace

import pytest
from PIL import Image

import backend
from backend import GenerationOptions, ModelManager, read_image


@pytest.mark.parametrize("changes", [
    {"prompt": " "}, {"width": 1000}, {"height": 4096},
    {"steps": 0}, {"seed": -2}, {"seed": 2**32},
])
def test_invalid_options(changes):
    with pytest.raises(ValueError):
        GenerationOptions(**({"prompt": "test"} | changes)).validate()


def test_read_image_preserves_palette_transparency(tmp_path):
    path = tmp_path / "palette.png"
    image = Image.new("P", (32, 32))
    image.save(path, transparency=0)
    loaded = read_image(path)
    assert loaded.mode == "RGBA"
    assert loaded.getpixel((0, 0))[3] == 0


def test_generation_keeps_reference_order_seed_and_rgba(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_DIR", tmp_path / "outputs")
    refs = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (32, 32), color).save(path)
        refs.append(str(path))
    captured = {}

    def pipeline(**kwargs):
        captured.update(kwargs)
        kwargs["callback_on_step_end"](None, 1, None, {})
        return SimpleNamespace(images=[Image.new("RGBA", (256, 256), (1, 2, 3, 0))])

    manager = ModelManager()
    monkeypatch.setattr(manager, "load_pipeline", lambda: pipeline)
    image_path, metadata_path, metadata = manager.generate(
        refs, GenerationOptions("Combine the subjects", 256, 256, 2, 42, True),
    )
    assert captured["generator"].initial_seed() == 42
    assert captured["image"][0].getpixel((0, 0)) == (255, 0, 0)
    assert captured["image"][1].getpixel((0, 0)) == (0, 0, 255)
    assert "alpha channel" in captured["prompt"]
    assert captured["true_cfg_scale"] == 1.0
    with Image.open(image_path) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
    assert json.loads(open(metadata_path).read()) == metadata
    assert metadata["task"] == "image-to-image"


def test_text_to_image_omits_references(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_DIR", tmp_path)
    manager = ModelManager()

    def pipeline(**kwargs):
        assert kwargs["image"] is None
        return SimpleNamespace(images=[Image.new("RGB", (256, 256))])

    monkeypatch.setattr(manager, "load_pipeline", lambda: pipeline)
    _, _, metadata = manager.generate([], GenerationOptions("test", 256, 256, 1))
    assert metadata["task"] == "text-to-image"
    assert 0 <= metadata["seed"] < 2**32


def test_reference_limit_before_model_loading():
    with pytest.raises(ValueError, match="10"):
        ModelManager().generate(["missing.png"] * 11, GenerationOptions("test"))


def test_unload_releases_all_models():
    manager = ModelManager()
    manager.pipeline = object()
    manager.unload()
    assert manager.pipeline is None


def test_unknown_device(monkeypatch):
    monkeypatch.setenv("QWEN_DEVICE", "typo")
    with pytest.raises(ValueError, match="QWEN_DEVICE"):
        ModelManager().select_device()
