"""Local Qwen Image 2.1 inference."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

# Set before importing torch; unsupported MPS operations can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from PIL import Image, ImageOps  # noqa: E402

from prompt_enhancer import PromptEnhancer  # noqa: E402

IMAGE_MODEL = "Qwen/Qwen-Image-2.1"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent / "outputs")).resolve()


@dataclass(frozen=True)
class GenerationOptions:
    prompt: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = 40
    seed: int = -1
    transparent: bool = False

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("指示を入力してください。")
        if not 1 <= self.steps <= 100:
            raise ValueError("ステップ数は 1〜100 にしてください。")
        for size in (self.width, self.height):
            if not 256 <= size <= 3072 or size % 32:
                raise ValueError("幅と高さは 256〜3072 の範囲で、32 の倍数にしてください。")
        if not -1 <= self.seed <= 2**32 - 1:
            raise ValueError("シードは -1（ランダム）または 0〜4294967295 にしてください。")


def read_image(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
        return image.convert(mode)


def save_result(image: Image.Image, metadata: dict) -> tuple[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    image_path = OUTPUT_DIR / f"{stem}.png"
    metadata_path = OUTPUT_DIR / f"{stem}.json"
    image.save(image_path, format="PNG")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(image_path), str(metadata_path)


class ModelManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.pipeline = None
        self.device = None
        self.enhancer = PromptEnhancer()
        self.rewrite_cache = {}
        self.rewrite_stats = {}

    def select_device(self) -> str:
        import torch

        requested = os.getenv("QWEN_DEVICE", "auto")
        if requested not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("QWEN_DEVICE は auto / cuda / mps / cpu から選んでください。")
        if requested == "auto":
            requested = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA が利用できません。QWEN_DEVICE=auto を指定してください。")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS が利用できません。QWEN_DEVICE=auto を指定してください。")
        self.device = requested
        return requested

    def _release(self) -> None:
        import torch

        self.pipeline = None
        self.enhancer.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def unload(self) -> str:
        with self.lock:
            self._release()
            self.rewrite_cache.clear()
        return "モデルを解放しました。次回の実行時に再読み込みします。"

    def rewrite_prompt(
        self, paths: list[str], prompt: str,
        progress: Callable = lambda *args, **kwargs: None,
        *, thorough: bool = False,
    ) -> str:
        if not prompt.strip():
            raise ValueError("編集指示を入力してください。")
        if not 1 <= len(paths) <= 10:
            raise ValueError("編集する参照画像を 1〜10 枚アップロードしてください。")
        images = [read_image(path) for path in paths]
        image_key = tuple(
            (image.mode, image.size, hashlib.sha256(image.tobytes()).digest()) for image in images
        )
        prompt = prompt.strip()
        cache_key = (image_key, prompt, thorough)
        with self.lock:
            started = time.perf_counter()
            if cache_key in self.rewrite_cache:
                self.rewrite_stats = {"cache_hit": True, "total_seconds": 0.0}
                progress(1, desc="同じ画像・指示の前回結果を再利用しました")
                return self.rewrite_cache[cache_key]
            # Keep only one large model in memory at a time, including on MPS.
            self._release()
            try:
                device = self.select_device()
                result = self.enhancer.rewrite(
                    images, prompt, device, thorough=thorough, progress=progress,
                )
                # Cache only the last request and its output, so repeated clicks are idempotent.
                self.rewrite_cache = {
                    cache_key: result, (image_key, result, thorough): result,
                }
                self.rewrite_stats = {
                    **self.enhancer.last_stats, "cache_hit": False,
                    "total_seconds": round(time.perf_counter() - started, 2),
                }
                progress(1, desc="編集指示を整えました")
                return result
            finally:
                self._release()

    def load_pipeline(self):
        if self.pipeline is None:
            import torch
            from diffusers import QwenImage21Pipeline

            self._release()
            device = self.select_device()
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            pipeline = QwenImage21Pipeline.from_pretrained(IMAGE_MODEL, dtype=dtype)
            if device == "cuda" and os.getenv("QWEN_CPU_OFFLOAD", "1") == "1":
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)
            pipeline.vae.enable_tiling()
            self.pipeline = pipeline
        return self.pipeline

    def generate(
        self, paths: list[str], options: GenerationOptions,
        progress: Callable = lambda *args, **kwargs: None,
    ) -> tuple[str, str, dict]:
        options.validate()
        if len(paths) > 10:
            raise ValueError("参照画像は最大 10 枚です。")
        images = [read_image(path) for path in paths]
        seed = secrets.randbelow(2**32) if options.seed == -1 else options.seed
        prompt = options.prompt.strip()
        if options.transparent:
            prompt = (
                f"This is an RGBA image with transparency. {prompt}. "
                "The image has alpha channel and the background is transparent."
            )
        with self.lock:
            import torch

            started = time.monotonic()
            progress(0, desc="モデルを読み込み中（初回はダウンロードします）")
            pipeline = self.load_pipeline()
            load_seconds = time.monotonic() - started
            inference_started = time.monotonic()
            progress(0.02, desc="指示・参照画像を処理中")

            def on_step_end(pipe, step, timestep, callback_kwargs):
                if step + 1 == options.steps:
                    progress(0.9, desc=f"生成 {options.steps}/{options.steps} · 画像を仕上げ中")
                else:
                    progress(0.05 + 0.85 * (step + 1) / options.steps,
                             desc=f"生成中 {step + 1}/{options.steps} ステップ")
                return callback_kwargs

            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=images or None,
                    width=options.width,
                    height=options.height,
                    num_inference_steps=options.steps,
                    true_cfg_scale=1.0,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    callback_on_step_end=on_step_end,
                ).images[0]
            inference_seconds = time.monotonic() - inference_started
            progress(0.95, desc="生成画像と設定を保存中")
            metadata = {
                "model": IMAGE_MODEL, "task": "image-to-image" if images else "text-to-image",
                "prompt": options.prompt, "effective_prompt": prompt, "seed": seed,
                "width": result.width, "height": result.height, "steps": options.steps,
                "mode": result.mode, "reference_count": len(images), "device": self.device,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "load_seconds": round(load_seconds, 2),
                "inference_seconds": round(inference_seconds, 2),
            }
            image_path, metadata_path = save_result(result, metadata)
            progress(1, desc="画像と設定の保存が完了しました")
            return image_path, metadata_path, metadata
