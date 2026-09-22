"""Local image editing prompt rewriting with Qwen's fine-tuned VLM."""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
logger = logging.getLogger(__name__)
FAST_PREFIX = '{"rewritten_prompt": "'
FAST_MAX_TOKENS = 768
FAST_MAX_SECONDS = 75
FAST_TOTAL_PIXELS = 1024 * 1024
FAST_SYSTEM_PROMPT = """Rewrite the user's image editing request as a concise, actionable
instruction for an image editor. Look at the supplied images and preserve the user's intent.
Lead with the requested change. Preserve all unspecified content, identity and composition.
Do not invent objects or add unrequested changes. Keep explicit text to render exactly as given,
including its original language. Write descriptive prose in English, or Chinese for Chinese input.
For multiple images, use <image1>, <image2>, etc. in upload order and specify their roles.
For a single image, refer to it naturally. Do not put resolution or aspect ratio in the instruction;
the application sets those separately. Use one concise paragraph, usually 60-120 words, and retain
all essential user constraints. Output only a JSON object with the string field rewritten_prompt.
Do not output analysis, reasoning, alternatives or explanations. Finish the JSON and stop."""


class IncompleteRewrite(ValueError):
    """The model did not produce a complete instruction within its budget."""


def fast_image_options(image_count: int) -> dict:
    # Qwen2VLImageProcessor applies per-call bounds only when both are supplied.
    return {"min_pixels": 65536, "max_pixels": max(65536, FAST_TOTAL_PIXELS // image_count)}


def parse_fast_prompt(text: str) -> str:
    """Read a complete first field even if unused trailing fields are incomplete."""
    prefix = '{"rewritten_prompt":'
    if not text.startswith(prefix):
        raise ValueError("Missing rewritten_prompt field")
    value = text[len(prefix):].lstrip()
    try:
        result, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as error:
        raise ValueError("Incomplete rewritten_prompt string") from error
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Empty rewritten_prompt string")
    if not value[end:].lstrip().startswith((",", "}")):
        raise ValueError("Missing field delimiter")
    return result.strip()


class RewriteMonitor:
    """Stop a completed answer and report progress during autoregressive decoding."""

    def __init__(self, tokenizer, input_length, thorough, progress):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.thorough = thorough
        self.progress = progress
        self.started = time.perf_counter()
        self.last_update = self.started
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        count = input_ids.shape[1] - self.input_length
        now = time.perf_counter()
        elapsed = now - self.started
        if now - self.last_update >= 1:
            self.progress(0.5, desc=f"編集指示を作成中 · {count} トークン · {elapsed:.0f} 秒")
            self.last_update = now
        if not self.thorough:
            if count % 8 == 0:
                text = self.tokenizer.decode(
                    input_ids[0, self.input_length:], skip_special_tokens=True,
                )
                try:
                    parse_fast_prompt(FAST_PREFIX + text)
                except ValueError:
                    pass
                else:
                    self.reason = "complete"
                    return True
            if elapsed >= FAST_MAX_SECONDS:
                self.reason = "time_budget"
                return True
        return False


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
        from transformers import StoppingCriteriaList

        self.last_stats = {}
        progress(0.05, desc="補助モデルを読み込み中（初回のみダウンロード）")
        started = time.perf_counter()
        self.load(device)
        load_seconds = time.perf_counter() - started
        messages = [
            {"role": "system", "content": [{
                "type": "text", "text": self.system_prompt if thorough else FAST_SYSTEM_PROMPT,
            }]},
            {"role": "user", "content": [
                *({"type": "image", "image": image.convert("RGB")} for image in images),
                {"type": "text", "text": prompt},
            ]},
        ]
        template_options = {"add_generation_prompt": True}
        if not thorough:
            # Prefill the answer so the model immediately writes the useful field.
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": FAST_PREFIX},
            ]})
            template_options = {
                "continue_final_message": True,
                "processor_kwargs": {"images_kwargs": fast_image_options(len(images))},
            }
        progress(0.15, desc="参照画像を読み取り中")
        inputs = self.processor.apply_chat_template(
            messages, **template_options, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=thorough,
        ).to(device)
        progress(0.25, desc="編集指示を詳しく検討中" if thorough else "編集指示を作成中")
        started = time.perf_counter()
        monitor = RewriteMonitor(
            self.processor.tokenizer, inputs["input_ids"].shape[1], thorough, progress,
        )
        generation_options = (
            {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            if thorough else {"do_sample": False}
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=24000 if thorough else FAST_MAX_TOKENS,
                **generation_options, stopping_criteria=StoppingCriteriaList([monitor]),
            )
        text = self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        self.last_stats = {
            "load_seconds": round(load_seconds, 2),
            "generate_seconds": round(time.perf_counter() - started, 2),
            "generated_tokens": generated_tokens,
            "thorough": thorough,
            "input_tokens": inputs["input_ids"].shape[1],
            "stop_reason": monitor.reason or (
                "token_limit" if generated_tokens >= (24000 if thorough else FAST_MAX_TOKENS)
                else "eos"
            ),
        }
        logger.info("Prompt enhancement timings: %s", self.last_stats)
        try:
            return parse_rewritten_prompt(text) if thorough else parse_fast_prompt(FAST_PREFIX + text)
        except ValueError as error:
            logger.warning("Incomplete prompt rewrite: %s; timings=%s", error, self.last_stats)
            raise IncompleteRewrite(
                "補助モデルが完成した指示文を返さなかったため、書き換えませんでした。"
                "元の指示はそのまま「画像を生成」に使えます。"
            ) from error
