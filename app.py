"""Japanese Gradio interface for local Qwen image workflows."""

import argparse
import logging
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from backend import DEFAULT_HEIGHT, DEFAULT_WIDTH, GenerationOptions, ModelManager  # noqa: E402

manager = ModelManager()
logger = logging.getLogger(__name__)


def handle_error(error: Exception):
    if isinstance(error, ValueError):
        raise gr.Error(str(error)) from error
    logger.exception("Inference failed")
    detail = str(error)
    if "out of memory" in detail.lower():
        detail = "メモリが不足しました。モデルを解放し、解像度・参照画像数を減らしてください。"
    raise gr.Error(f"実行に失敗しました: {detail}") from error


def generate_image(
    task, references, prompt, width, height, steps, seed, transparent,
    progress=gr.Progress(),
):
    try:
        paths = references or []
        if task == "Image to Image" and not paths:
            raise ValueError("編集する参照画像を 1 枚以上アップロードしてください。")
        if task == "Text to Image":
            paths = []
        options = GenerationOptions(
            prompt, int(width), int(height), int(steps), int(seed), transparent,
        )
        image_path, metadata_path, metadata = manager.generate(paths, options, progress)
        return image_path, [image_path, metadata_path], metadata
    except Exception as error:
        handle_error(error)


def build_app():
    with gr.Blocks(title="Image Workbench", delete_cache=(3600, 86400)) as demo:
        gr.Markdown(
            "# Image Workbench\n"
            "Qwen Image 2.1 で、画像を編集・文章から画像を生成。\n\n"
            "ローカル推論 · 初回実行時にモデルをダウンロードします。"
        )
        with gr.Tab("画像編集・生成"):
            with gr.Row():
                with gr.Column():
                    task = gr.Radio(
                        ["Image to Image", "Text to Image"],
                        value="Image to Image", label="モード",
                    )
                    references = gr.File(
                        file_count="multiple", file_types=["image"],
                        label="参照画像（最大 10 枚・番号はアップロード順）",
                    )
                    preview = gr.Gallery(
                        label="参照画像プレビュー", columns=3, height=180,
                        interactive=False, format="png", visible=False,
                    )
                    prompt = gr.Textbox(
                        label="編集・生成の指示", lines=4,
                        placeholder="例：人物の顔と服を保ったまま、背景を夕暮れの海辺に変えてください。",
                    )
                    gr.Examples(
                        examples=[
                            ["背景を夕暮れの海辺に変更。人物の顔・服・構図は維持してください。"],
                            ["Extract the main subject and make the background transparent."],
                            ["A small ceramic fox on a wooden desk, soft morning light."],
                        ], inputs=prompt,
                    )
                    with gr.Accordion("生成設定", open=True):
                        with gr.Row():
                            width = gr.Slider(
                                256, 3072, value=DEFAULT_WIDTH, step=32, label="幅 (px)",
                                min_width=240,
                            )
                            height = gr.Slider(
                                256, 3072, value=DEFAULT_HEIGHT, step=32, label="高さ (px)",
                                min_width=240,
                            )
                        steps = gr.Slider(1, 100, value=40, step=1, label="ステップ数")
                        seed = gr.Number(value=-1, precision=0, label="シード（-1: ランダム）")
                        transparent = gr.Checkbox(label="透明背景を指示する（RGBA）")
                        gr.Markdown(
                            f"初期サイズは {DEFAULT_WIDTH}×{DEFAULT_HEIGHT} px（幅×高さ）です。"
                            "大きい画像は時間とメモリを多く使います。"
                        )
                    run = gr.Button("画像を生成", variant="primary")
                with gr.Column():
                    output = gr.Image(
                        label="生成結果", type="filepath", format="png", image_mode="RGBA",
                        interactive=False, height=520,
                    )
                    files = gr.File(label="PNG・生成設定をダウンロード", file_count="multiple")
                    with gr.Accordion("生成情報", open=False):
                        info = gr.JSON(label="設定と実行結果")
            task.change(
                lambda mode, paths: (
                    gr.File(visible=mode == "Image to Image"),
                    gr.Gallery(visible=mode == "Image to Image" and bool(paths)),
                ),
                [task, references], [references, preview], queue=False,
            )
            references.change(
                lambda paths, mode: gr.Gallery(
                    value=[(path, f"Image {i + 1}") for i, path in enumerate(paths or [])],
                    visible=mode == "Image to Image" and bool(paths),
                ),
                [references, task], preview, queue=False,
            )
            run.click(
                generate_image,
                [task, references, prompt, width, height, steps, seed, transparent],
                [output, files, info], concurrency_limit=1, concurrency_id="inference",
                api_name="generate_image",
            )
        with gr.Accordion("モデル・保存先", open=False):
            gr.Markdown(
                "生成した PNG と設定 JSON は `outputs/` に保存されます。\n\n"
                "モデル: [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)"
            )
            unload = gr.Button("モデルを解放")
            status = gr.Textbox(label="状態", interactive=False)
            unload.click(manager.unload, outputs=status, concurrency_id="inference")
    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local image workbench")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_app().launch(
        server_name=args.host, server_port=args.port, share=False,
        theme=gr.themes.Soft(primary_hue="teal"),
    )
