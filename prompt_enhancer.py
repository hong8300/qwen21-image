"""Local image editing prompt rewriting with Qwen's fine-tuned VLM."""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
logger = logging.getLogger(__name__)


def parse_rewritten_prompt(text: str) -> str:
    """Extract the final JSON answer, without exposing the reasoning block."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("補助モデルの回答が途中で終了しました。もう一度お試しください。")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("補助モデルの回答を読み取れませんでした。もう一度お試しください。") from error
    prompt = result.get("rewritten_prompt") if isinstance(result, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("補助モデルから編集指示を取得できませんでした。もう一度お試しください。")
    return prompt.strip()


class PromptEnhancer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.system_prompt = None
        self.last_stats = {}

    def unload(self):
        self.model = None
        self.processor = None
        self.system_prompt = None

    def load(self, device: str):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        prompt_path = hf_hub_download(MODEL_ID, "system_prompt.txt")
        self.system_prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype)
        self.model.to(device).eval()

    def rewrite(
        self, images, prompt: str, device: str, *, thorough: bool = False,
        progress: Callable = lambda *args, **kwargs: None,
    ) -> str:
        import torch

        self.last_stats = {}
        progress(0.05, desc="補助モデルを読み込み中（初回のみダウンロード）")
        started = time.perf_counter()
        self.load(device)
        load_seconds = time.perf_counter() - started
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                *({"type": "image", "image": image.convert("RGB")} for image in images),
                {"type": "text", "text": prompt},
            ]},
        ]
        progress(0.15, desc="参照画像を読み取り中")
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=thorough,
        ).to(device)
        progress(0.25, desc="編集指示を詳しく検討中" if thorough else "編集指示を作成中")
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=24000 if thorough else 2048,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
            )
        text = self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        self.last_stats = {
            "load_seconds": round(load_seconds, 2),
            "generate_seconds": round(time.perf_counter() - started, 2),
            "generated_tokens": output.shape[1] - inputs["input_ids"].shape[1],
            "thorough": thorough,
        }
        logger.info("Prompt enhancement timings: %s", self.last_stats)
        try:
            return parse_rewritten_prompt(text)
        except ValueError as error:
            if thorough:
                raise
            raise ValueError(
                "通常モードで編集指示を取得できませんでした。"
                "「詳しく検討する」をオンにして再実行してください。"
            ) from error
