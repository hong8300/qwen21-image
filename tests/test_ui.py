"""Reference replacement and user-visible action states."""

import pytest

import app
from ui import action_availability, combine_references, status_card


def test_reference_replacement_addition_and_clear_keep_order():
    assert combine_references(["new.png"], ["old.png"], "入れ替え") == ["new.png"]
    assert combine_references(["new.png"], ["old.png"], "追加") == ["old.png", "new.png"]
    assert combine_references([], ["old.png"], "入れ替え") == ["old.png"]
    assert app.accept_references(["new.png"], ["old.png"], "入れ替え") == (["new.png"], None)


def test_excess_images_leave_old_references_and_clear_drop_zone(monkeypatch):
    warnings = []
    monkeypatch.setattr(app.gr, "Warning", lambda message: warnings.append(message))
    existing = [f"{i}.png" for i in range(10)]
    paths, uploader = app.accept_references(["new.png"], existing, "追加")
    assert paths == existing and uploader is None
    assert warnings and "10枚" in warnings[0]


def test_generated_image_can_become_next_edit_input():
    assert app.reuse_generated_image("output.png") == (["output.png"], "Image to Image", None)
    with pytest.raises(app.gr.Error):
        app.reuse_generated_image(None)


@pytest.mark.parametrize("mode,refs,prompt,generate,enhance", [
    ("Image to Image", [], "", False, False),
    ("Image to Image", [], "Edit", False, False),
    ("Image to Image", ["ref.png"], " ", False, False),
    ("Image to Image", ["ref.png"], "Edit", True, True),
    ("Image to Image", ["ref.png"] * 11, "Edit", False, False),
    ("Text to Image", [], "Create", True, False),
    ("Text to Image", [], "", False, False),
])
def test_actions_require_valid_inputs(mode, refs, prompt, generate, enhance):
    allowed = action_availability(mode, refs, prompt, None)
    assert allowed["generate"] is generate and allowed["enhance"] is enhance
    assert allowed["combined"] is enhance
    assert allowed["reason"]


def test_all_mutating_actions_are_blocked_during_inference_and_restored_afterwards():
    args = ("Image to Image", ["ref.png"], "Edit", "output.png")
    busy = action_availability(*args, busy=True)
    ready = action_availability(*args, busy=False)
    for key in ["generate", "enhance", "combined", "clear", "reuse", "edit"]:
        assert busy[key] is False and ready[key] is True


def test_status_is_textual_as_well_as_colored_and_escapes_errors():
    assert "現在生成中" in status_card("running", "Loading", "generate")
    assert "現在指示を整理中" in status_card("running", "Loading", "enhance")
    error = status_card("error", '<script>alert("test")</script>')
    assert '<script>' not in error and '&lt;script&gt;' in error
    assert 'role="status"' in error and 'data-state="error"' in error
