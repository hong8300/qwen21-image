"""Per-execution progress, elapsed time and logging for streaming UI responses."""

import contextvars
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("workbench")
_current_run = contextvars.ContextVar("current_run", default=None)
_setup_lock = threading.Lock()
_configured = False


class KnownFallbackFilter(logging.Filter):
    """Hide only the four expected optional-kernel warnings, not other failures."""

    pattern = re.compile(
        r"^`(?:causal_conv1d_fn|causal_conv1d_update|chunk_gated_delta_rule|"
        r"fused_recurrent_gated_delta_rule)` is falling back to its reference PyTorch "
        r"implementation because `(?:causal_conv1d|flash-linear-attention)` is not installed\. "
        r"This is correct but much slower; install `(?:causal_conv1d|flash-linear-attention)` "
        r"for the optimized kernel\.$"
    )

    def filter(self, record):
        return not (
            record.levelno == logging.WARNING and self.pattern.fullmatch(record.getMessage())
        )


class ExecutionLogHandler(logging.Handler):
    def emit(self, record):
        run = _current_run.get()
        if run is not None and not getattr(record, "_workbench_captured", False):
            record._workbench_captured = True
            run.append_log(self.format(record))


def configure_logging():
    global _configured
    with _setup_lock:
        if _configured:
            return
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
        logger.setLevel(logging.INFO)
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger("transformers.integrations.hub_kernels").addFilter(KnownFallbackFilter())
        handler = ExecutionLogHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        # These libraries may disable propagation and install their own console handlers.
        for name in ("", "transformers", "diffusers"):
            logging.getLogger(name).addHandler(handler)
        logging.captureWarnings(True)
        _configured = True


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)} 分 {remaining:.1f} 秒"


@dataclass(frozen=True)
class RunUpdate:
    status: str
    log: str
    elapsed: float
    done: bool
    result: object = None
    error: Exception | None = None


class Execution:
    def __init__(self, label):
        self.label = label
        self.started = time.monotonic()
        self.finished = None
        self.stage = "準備中"
        self.lines = deque(maxlen=200)
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.result = None
        self.error = None

    def append_log(self, message):
        with self.lock:
            self.lines.extend(line[:4000] for line in message.splitlines())

    def progress(self, value=None, desc=None, **kwargs):
        if desc:
            with self.lock:
                changed = desc != self.stage
                self.stage = desc
            if changed:
                logger.info("%s: %s", self.label, desc)

    def snapshot(self):
        with self.lock:
            elapsed = (
                self.finished if self.finished is not None else time.monotonic()
            ) - self.started
            return RunUpdate(
                status=f"{self.label} · {self.stage} · 経過 {format_seconds(elapsed)}",
                log="\n".join(self.lines), elapsed=elapsed, done=self.done.is_set(),
                result=self.result, error=self.error,
            )

    def run(self, operation):
        token = _current_run.set(self)
        try:
            logger.info("%s: 開始", self.label)
            self.result = operation(self.progress)
            self.stage = "完了"
            logger.info("%s: 完了 · 所要時間 %s", self.label,
                        format_seconds(time.monotonic() - self.started))
        except Exception as error:
            self.error = error
            self.stage = "終了（詳細は実行ログ）"
            logger.exception("%s: 完了できませんでした · %s", self.label, error)
        finally:
            with self.lock:
                self.finished = time.monotonic()
                self.done.set()
            _current_run.reset(token)


def stream_execution(label, operation, interval=0.5):
    """Keep the UI clock live even when model loading has no progress callback."""
    configure_logging()
    run = Execution(label)
    worker = threading.Thread(target=run.run, args=(operation,), daemon=True)
    worker.start()
    try:
        while not run.done.is_set():
            yield run.snapshot()
            run.done.wait(interval)
        yield run.snapshot()
    finally:
        # Keep the inference slot occupied if the browser disconnects mid-operation.
        worker.join()
