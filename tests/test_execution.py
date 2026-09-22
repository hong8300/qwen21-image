"""Progress streaming, log isolation, and targeted console filtering."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import app
from execution import KnownFallbackFilter, configure_logging, format_seconds, stream_execution
from prompt_enhancer import IncompleteRewrite


@pytest.mark.parametrize("function,package", [
    ("causal_conv1d_fn", "causal_conv1d"),
    ("causal_conv1d_update", "causal_conv1d"),
    ("chunk_gated_delta_rule", "flash-linear-attention"),
    ("fused_recurrent_gated_delta_rule", "flash-linear-attention"),
])
def test_only_known_fallback_warnings_are_hidden(function, package):
    message = (
        f"`{function}` is falling back to its reference PyTorch implementation because "
        f"`{package}` is not installed. This is correct but much slower; install "
        f"`{package}` for the optimized kernel."
    )
    record = logging.LogRecord("transformers.integrations.hub_kernels", logging.WARNING,
                               __file__, 1, message, (), None)
    filter_ = KnownFallbackFilter()
    assert filter_.filter(record) is False
    record.levelno = logging.ERROR
    assert filter_.filter(record) is True
    record.levelno = logging.WARNING
    record.msg = "An unexpected kernel warning"
    assert filter_.filter(record) is True


def test_http_info_is_quiet_but_warnings_are_retained():
    configure_logging()
    http = logging.getLogger("httpx")
    assert not http.isEnabledFor(logging.INFO)
    assert http.isEnabledFor(logging.WARNING)


def test_elapsed_time_updates_during_silent_loading_and_finishes():
    release = threading.Event()
    entered = threading.Event()

    def operation(progress):
        progress(0, desc="モデルを読み込み中")
        entered.set()
        assert release.wait(5)
        progress(1, desc="保存完了")
        return "image.png"

    stream = stream_execution("生成", operation, interval=0.01)
    first = next(stream)
    assert entered.wait(5)
    second = next(stream)
    assert not second.done and second.elapsed > first.elapsed
    assert "モデルを読み込み中" in second.status
    release.set()
    final = list(stream)[-1]
    assert final.done and final.result == "image.png" and final.error is None
    assert "保存完了" in final.log and "所要時間" in final.log


def test_logs_are_isolated_between_concurrent_executions_and_bounded():
    barrier = threading.Barrier(2)

    def consume(name):
        def operation(progress):
            barrier.wait(timeout=5)
            logging.getLogger("workbench").warning("Only %s", name)
            return name
        return list(stream_execution(name, operation))[-1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(consume, ["session-A", "session-B"])
    assert "Only session-A" in a.log and "session-B" not in a.log
    assert "Only session-B" in b.log and "session-A" not in b.log

    def noisy(progress):
        for i in range(250):
            logging.getLogger("workbench").info("line %s", i)
    final = list(stream_execution("bounded", noisy))[-1]
    assert len(final.log.splitlines()) == 200
    assert "line 249" in final.log and "line 0\n" not in final.log


def test_disconnect_keeps_worker_accounted_for_until_it_finishes():
    release = threading.Event()
    closed = threading.Event()

    def operation(progress):
        assert release.wait(5)

    stream = stream_execution("disconnect", operation)
    next(stream)

    def close():
        stream.close()
        closed.set()
    closer = threading.Thread(target=close)
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    closer.join(timeout=5)
    assert closed.is_set()


@pytest.mark.parametrize("cache_hit", [False, True])
def test_enhancement_stream_has_separate_duration_and_cache_result(monkeypatch, cache_hit):
    def rewrite(paths, prompt, progress, **kwargs):
        progress(0.5, desc="16 トークン")
        app.manager.rewrite_stats = {
            "cache_hit": cache_hit, "load_seconds": 1.0, "generate_seconds": 2.0,
        }
        return "Rewritten"

    monkeypatch.setattr(app.manager, "rewrite_prompt", rewrite)
    final = list(app.stream_enhance_prompt("Image to Image", ["image.png"], "Original"))[-1]
    assert final[0] == "Rewritten"
    assert "完了" in final[1] and "完了" in final[2]
    assert ("前回の結果を再利用" in final[2]) is cache_hit
    assert "16 トークン" in final[3]


def test_incomplete_enhancement_stream_keeps_original_and_shows_warning(monkeypatch):
    warnings = []

    def incomplete(*args, **kwargs):
        raise IncompleteRewrite("回答が未完成")

    monkeypatch.setattr(app.manager, "rewrite_prompt", incomplete)
    monkeypatch.setattr(app.gr, "Warning", lambda text, **kwargs: warnings.append(text))
    final = list(app.stream_enhance_prompt("Image to Image", ["image.png"], "Original"))[-1]
    assert final[0] == app.gr.skip()
    assert "未完了・元の指示を保持" in final[2]
    assert "回答が未完成" in final[3] and warnings == ["回答が未完成"]


def test_generation_failure_retains_error_log_and_elapsed_time(monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError("device failure")

    monkeypatch.setattr(app, "generate_image", failing)
    stream = app.stream_generate_image("Text to Image", [], "test", 256, 256, 1, 42, False)
    updates = []
    with pytest.raises(app.gr.Error, match="device failure"):
        for update in stream:
            updates.append(update)
    assert updates[-1][:3] == (app.gr.skip(),) * 3
    assert "失敗" in updates[-1][4]
    assert "device failure" in updates[-1][5]


def test_generation_success_displays_load_and_inference_times(monkeypatch):
    metadata = {"load_seconds": 1.5, "inference_seconds": 4.5}
    monkeypatch.setattr(app, "generate_image", lambda *args: ("image.png", [], metadata))
    final = list(app.stream_generate_image(
        "Text to Image", [], "test", 256, 256, 1, 42, False,
    ))[-1]
    assert final[:3] == ("image.png", [], metadata)
    assert "読込 1.50 秒 / 生成 4.50 秒" in final[4]


def test_duration_formats_minutes_and_fast_cache_hits():
    assert format_seconds(0.03) == "0.03 秒"
    assert format_seconds(123.4) == "2 分 3.4 秒"
