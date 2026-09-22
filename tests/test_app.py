"""Check size settings from the UI through inference and PNG metadata."""

from types import SimpleNamespace

import pytest
from PIL import Image

import app
import backend


def test_preview_tracks_upload_order_clear_and_mode_changes():
    paths = ["portrait.png", "landscape.png"]
    shown = app.update_reference_preview(paths, "Image to Image")
    assert shown["value"] == [(paths[0], "Image 1"), (paths[1], "Image 2")]
    assert shown["selected_index"] == 0 and shown["visible"] is True
    assert app.update_reference_preview(paths, "Text to Image")["visible"] is False
    cleared = app.update_reference_preview([], "Image to Image")
    assert cleared["value"] == [] and cleared["selected_index"] is None
    assert cleared["visible"] is True


@pytest.mark.parametrize("task", ["Text to Image", "Image to Image"])
@pytest.mark.parametrize("size", [None, (768, 1024)])
def test_ui_size_reaches_pipeline_and_saved_image(tmp_path, monkeypatch, task, size):
    demo = app.build_app()
    sliders = {
        component["props"]["label"]: component["props"]["value"]
        for component in demo.config["components"] if component["type"] == "slider"
    }
    defaults = backend.GenerationOptions("test")
    assert (sliders["幅 (px)"], sliders["高さ (px)"]) == (
        defaults.width, defaults.height,
    ) == (1024, 768)
    width, height = size or (sliders["幅 (px)"], sliders["高さ (px)"])
    references = []
    if task == "Image to Image":
        reference = tmp_path / "reference.png"
        Image.new("RGB", (512, 512)).save(reference)
        references.append(str(reference))

    def pipeline(**kwargs):
        assert (kwargs["width"], kwargs["height"]) == (width, height)
        assert bool(kwargs["image"]) == bool(references)
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])

    manager = backend.ModelManager()
    monkeypatch.setattr(manager, "load_pipeline", lambda: pipeline)
    monkeypatch.setattr(app, "manager", manager)
    monkeypatch.setattr(backend, "OUTPUT_DIR", tmp_path / "outputs")
    image_path, _, metadata = app.generate_image(
        task, references, "test", width, height, 1, 42, False,
        progress=lambda *args, **kwargs: None,
    )
    with Image.open(image_path) as image:
        assert image.size == (width, height)
    assert (metadata["width"], metadata["height"]) == (width, height)
