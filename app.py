"""Japanese Gradio interface for local Qwen image workflows."""

import argparse
import logging
import os
import threading
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from backend import DEFAULT_HEIGHT, DEFAULT_WIDTH, GenerationOptions, ModelManager  # noqa: E402
from execution import configure_logging, format_seconds, stream_execution  # noqa: E402
from prompt_enhancer import IncompleteRewrite  # noqa: E402
from ui import (  # noqa: E402
    SCROLL_TO_LATEST, UI_CSS, UI_JS, action_availability, combine_references, status_card,
)

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


def enhance_prompt(task, references, prompt, thorough=False, progress=gr.Progress()):
    try:
        return rewrite_instruction(task, references, prompt, thorough, progress)
    except IncompleteRewrite as error:
        gr.Warning(str(error), duration=12)
        return gr.skip()
    except Exception as error:
        handle_error(error)


def rewrite_instruction(task, references, prompt, thorough, progress):
    if task != "Image to Image":
        raise ValueError("編集指示の補助は Image to Image モードで利用してください。")
    return manager.rewrite_prompt(references or [], prompt, progress, thorough=thorough)


def stream_generate_image(task, references, prompt, width, height, steps, seed, transparent):
    def operation(progress):
        return generate_image(
            task, references, prompt, width, height, steps, seed, transparent, progress,
        )

    for update in stream_execution("画像生成", operation):
        result = (gr.skip(), gr.skip(), gr.skip())
        duration = gr.skip()
        status = update.status
        if update.done:
            duration = f"{'失敗' if update.error else '完了'} · {format_seconds(update.elapsed)}"
            if update.error:
                status = f"画像生成に失敗: {update.error} · {format_seconds(update.elapsed)}"
            else:
                result = update.result
                metadata = result[2]
                duration += (
                    f"（読込 {format_seconds(metadata.get('load_seconds', 0))} / "
                    f"生成 {format_seconds(metadata.get('inference_seconds', 0))}）"
                )
        state = ("error" if update.error else "success") if update.done else "running"
        yield *result, status_card(state, status, "generate"), duration, update.log
    if update.error:
        raise gr.Error(str(update.error))


def stream_enhance_prompt(task, references, prompt, thorough=False):
    def operation(progress):
        with manager.lock:
            rewritten = rewrite_instruction(task, references, prompt, thorough, progress)
            return rewritten, dict(manager.rewrite_stats)

    for update in stream_execution("編集指示を整える", operation):
        result, duration = gr.skip(), gr.skip()
        status = update.status
        if update.done:
            if update.error:
                outcome = "未完了・元の指示を保持" if isinstance(
                    update.error, IncompleteRewrite,
                ) else "失敗"
                duration = f"{outcome} · {format_seconds(update.elapsed)}"
                status = f"編集指示を整える: {duration} · {update.error}"
            else:
                result, stats = update.result
                duration = f"完了 · {format_seconds(update.elapsed)}"
                if stats.get("cache_hit"):
                    duration += "（前回の結果を再利用）"
                else:
                    duration += (
                        f"（読込 {format_seconds(stats.get('load_seconds', 0))} / "
                        f"指示作成 {format_seconds(stats.get('generate_seconds', 0))}）"
                    )
        state = "running"
        if update.done:
            state = "warning" if isinstance(update.error, IncompleteRewrite) else (
                "error" if update.error else "success"
            )
        yield result, status_card(state, status, "enhance"), duration, update.log
    if isinstance(update.error, IncompleteRewrite):
        gr.Warning(str(update.error), duration=12)
    elif update.error:
        raise gr.Error(str(update.error))


def stream_enhance_and_generate(
    task, references, prompt, width, height, steps, seed, transparent, thorough=False,
):
    phase_lock = threading.Lock()
    phase = {"name": "enhance", "started": None, "prompt": gr.skip(),
             "enhancement_time": "処理中", "generation_time": "指示整理の完了待ち"}

    def operation(progress):
        with manager.lock:
            started = time.monotonic()
            with phase_lock:
                phase["started"] = started
            progress(0, desc="1/2 編集指示を整えています")
            rewritten = rewrite_instruction(
                task, references, prompt, thorough,
                lambda value=None, desc=None, **kw: progress(value, desc=f"1/2 指示整理 · {desc or ''}"),
            )
            stats = dict(manager.rewrite_stats)
            duration = f"完了 · {format_seconds(time.monotonic() - started)}"
            duration += "（前回の結果を再利用）" if stats.get("cache_hit") else (
                f"（読込 {format_seconds(stats.get('load_seconds', 0))} / "
                f"指示作成 {format_seconds(stats.get('generate_seconds', 0))}）"
            )
            logger.info("1/2 編集指示: %s", duration)
            with phase_lock:
                phase.update(name="generate", started=time.monotonic(), prompt=rewritten,
                             enhancement_time=duration, generation_time="処理中")
            progress(0, desc="2/2 整えた指示で画像を生成しています")
            result = generate_image(
                task, references, rewritten, width, height, steps, seed, transparent,
                lambda value=None, desc=None, **kw: progress(value, desc=f"2/2 画像生成 · {desc or ''}"),
            )
            metadata = result[2]
            with phase_lock:
                phase["generation_time"] = (
                    f"完了 · {format_seconds(time.monotonic() - phase['started'])}"
                    f"（読込 {format_seconds(metadata.get('load_seconds', 0))} / "
                    f"生成 {format_seconds(metadata.get('inference_seconds', 0))}）"
                )
            return result

    for update in stream_execution("指示整理 → 画像生成", operation):
        with phase_lock:
            current = dict(phase)
        state, status = "running", update.status
        result = (gr.skip(),) * 3
        incomplete = isinstance(update.error, IncompleteRewrite)
        if update.done:
            state = "warning" if incomplete else ("error" if update.error else "success")
            if update.error:
                duration = format_seconds(time.monotonic() - current["started"]) if current[
                    "started"
                ] is not None else format_seconds(update.elapsed)
                if current["name"] == "enhance":
                    current["enhancement_time"] = f"{'未完了・元の指示を保持' if incomplete else '失敗'} · {duration}"
                    current["generation_time"] = "未実行（指示整理で停止）"
                else:
                    current["generation_time"] = f"失敗 · {duration}"
                status = f"処理を停止: {update.error} · 合計 {format_seconds(update.elapsed)}"
            else:
                result = update.result
                status = f"指示整理・画像生成が完了 · 合計 {format_seconds(update.elapsed)}"
        yield (
            current["prompt"], *result,
            status_card(state, status, "combined-" + current["name"]),
            current["generation_time"], current["enhancement_time"], update.log,
        )
    if isinstance(update.error, IncompleteRewrite):
        gr.Warning(str(update.error), duration=12)
    elif update.error:
        raise gr.Error(str(update.error))


def update_reference_preview(paths, mode):
    return gr.update(
        value=[(path, f"Image {i + 1}") for i, path in enumerate(paths or [])],
        visible=mode == "Image to Image", selected_index=0 if paths else None,
        preview=True,
    )


def accept_references(uploaded, current, mode):
    try:
        paths = combine_references(uploaded, current, mode)
    except ValueError as error:
        gr.Warning(str(error))
        paths = list(current or [])
    # Keep the drop zone empty and available for the next upload.
    return paths, None


def reuse_generated_image(path):
    if not path:
        raise gr.Error("先に画像を生成してください。")
    return [path], "Image to Image", None


def build_app():
    configure_logging()
    with gr.Blocks(title="Image Workbench", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# Image Workbench\nQwen Image 2.1 · Macでローカル画像編集・生成", elem_id="app-heading")
        with gr.Column(elem_id="editor"):
            busy = gr.State(False)
            active_operation = gr.State("")
            references = gr.File(file_count="multiple", visible=False, interactive=False)
            with gr.Row(elem_id="workspace"):
                with gr.Column(min_width=320):
                    task = gr.Radio(
                        ["Image to Image", "Text to Image"],
                        value="Image to Image", label="モード",
                    )
                    with gr.Group() as reference_controls:
                        with gr.Row():
                            with gr.Column(min_width=170, scale=1):
                                upload_mode = gr.Radio(
                                    ["入れ替え", "追加"], value="入れ替え", label="画像の取り込み方",
                                    elem_id="upload-mode",
                                )
                                uploader = gr.File(
                                    file_count="multiple", file_types=["image"], interactive=True,
                                    label="参照画像をドロップ（最大10枚）",
                                    height=120, elem_id="reference-upload",
                                )
                            preview = gr.Gallery(
                                label="参照画像", columns=3, rows=1, height=210, min_width=170,
                                scale=1, interactive=False, format="png", object_fit="contain",
                                preview=True, selected_index=0, buttons=["fullscreen"],
                                elem_id="reference-preview",
                            )
                        with gr.Row():
                            reference_count = gr.Markdown("参照画像: **0 / 10枚**", elem_id="reference-count")
                            clear_references = gr.Button(
                                "参照画像をクリア", interactive=False, elem_classes="wb-action",
                                elem_id="clear-references", size="sm", min_width=140,
                            )
                    prompt = gr.Textbox(
                        label="編集・生成の指示", lines=3,
                        placeholder="例：顔と服を保ち、背景を夕暮れの海辺に変更。",
                        elem_id="instruction",
                    )
                    action_hint = gr.Markdown(
                        action_availability("Image to Image", [], "", None)["reason"],
                        elem_id="action-hint",
                    )
                    with gr.Row():
                        run = gr.Button(
                            "画像を生成", variant="primary", interactive=False,
                            elem_classes="wb-action", elem_id="generate-button", min_width=140,
                        )
                        combined = gr.Button(
                            "指示を整えてから生成", variant="primary", interactive=False,
                            elem_classes="wb-action", elem_id="combined-button", min_width=200,
                        )
                    with gr.Row() as enhancement_controls:
                        enhance = gr.Button(
                            "編集指示を整える", interactive=False, elem_classes="wb-action",
                            elem_id="enhance-button", min_width=170, size="sm",
                        )
                        thorough = gr.Checkbox(
                            value=False, label="詳しく検討する（時間がかかります）", min_width=200,
                        )
                    with gr.Accordion("生成設定 · サイズ・ステップ・シード", open=False):
                        with gr.Row():
                            width = gr.Slider(
                                256, 3072, value=DEFAULT_WIDTH, step=32, label="幅 (px)",
                                min_width=170,
                            )
                            height = gr.Slider(
                                256, 3072, value=DEFAULT_HEIGHT, step=32, label="高さ (px)",
                                min_width=170,
                            )
                        swap = gr.Button("幅と高さを入れ替える", elem_classes="wb-action")
                        with gr.Row():
                            steps = gr.Slider(1, 100, value=40, step=1, label="ステップ数", min_width=170)
                            seed = gr.Number(value=-1, precision=0, label="シード（-1: ランダム）", min_width=170)
                        transparent = gr.Checkbox(label="透明背景を指示する（RGBA）")
                        gr.Markdown(
                            f"初期サイズは {DEFAULT_WIDTH}×{DEFAULT_HEIGHT} px（幅×高さ）です。"
                            "大きい画像は時間とメモリを多く使います。"
                        )
                    with gr.Accordion("Examples · 指示の例", open=False, elem_id="examples") as example_controls:
                        gr.Examples(
                            examples=[
                                ["背景を夕暮れの海辺に変更。人物の顔・服・構図は維持してください。"],
                                ["Extract the main subject and make the background transparent."],
                                ["A small ceramic fox on a wooden desk, soft morning light."],
                            ], inputs=prompt,
                        )
                    with gr.Accordion("使い方・指示整理について", open=False):
                        gr.Markdown(
                            "**画像を生成**：入力した指示をそのまま使います。\n\n"
                            "**指示を整えてから生成**：参照画像を見て指示を書き直し、自動で生成します。"
                            "指示が未完成の場合は停止します。\n\n"
                            "**編集指示を整える**：指示文だけを書き直します。確認・修正してから生成できます。\n\n"
                            "通常は簡潔な指示を作ります。細部は「詳しく検討する」で検討できます。"
                            "同じ画像・指示は前回の結果を再利用します。初回のみ補助モデル約18.8 GBをダウンロードします。"
                        )
                with gr.Column(min_width=320):
                    run_status = gr.HTML(
                        status_card("idle", "参照画像と指示を準備してください。"),
                        elem_id="run-status",
                    )
                    output = gr.Image(
                        label="生成結果", type="filepath", format="png", image_mode="RGBA",
                        interactive=False, height=360,
                        buttons=["download", "fullscreen"], elem_id="generated-image",
                    )
                    reuse_result = gr.Button(
                        "この生成画像を参照画像にする", interactive=False,
                        elem_classes="wb-action", elem_id="reuse-result",
                    )
                    with gr.Row():
                        generation_time = gr.Textbox(
                            label="画像生成の所要時間（直近）", value="未実行", interactive=False,
                            elem_id="generation-time", min_width=170,
                        )
                        enhancement_time = gr.Textbox(
                            label="指示を整える所要時間（直近）", value="未実行", interactive=False,
                            elem_id="enhancement-time", min_width=170,
                        )
                    with gr.Accordion("実行ログ", open=False):
                        with gr.Row():
                            gr.Checkbox(
                                label="最新行へ自動スクロール", value=True, interactive=True,
                                elem_id="follow-log",
                            )
                            latest_log = gr.Button(
                                "最新行へ", elem_id="latest-log", size="sm", elem_classes="wb-action",
                            )
                        run_log = gr.Textbox(
                            label="ログ（最新200行）", lines=12, max_lines=12, interactive=False,
                            autoscroll=False, buttons=["copy"], elem_id="run-log",
                        )
                    with gr.Accordion("ダウンロード・生成情報", open=False):
                        files = gr.File(label="PNG・生成設定をダウンロード", file_count="multiple")
                        info = gr.JSON(label="設定と実行結果")
            task.change(
                lambda mode, paths: (
                    gr.update(visible=mode == "Image to Image"),
                    gr.update(visible=mode == "Image to Image"),
                    update_reference_preview(paths, mode),
                ),
                [task, references], [reference_controls, enhancement_controls, preview], queue=False,
                api_name=False,
            )
            references.change(
                lambda paths, mode: (
                    update_reference_preview(paths, mode),
                    f"参照画像: **{len(paths or [])} / 10枚**",
                ),
                [references, task], [preview, reference_count], queue=False, api_name=False,
            )
            uploader.upload(
                accept_references, [uploader, references, upload_mode], [references, uploader],
                queue=False, api_name="set_references",
            )
            clear_references.click(
                lambda: ([], None), outputs=[references, uploader], queue=False, api_name=False,
            )
            reuse_result.click(
                reuse_generated_image, output, [references, task, uploader],
                queue=False, api_name="reuse_result",
            )
            latest_log.click(None, js=SCROLL_TO_LATEST, queue=False, api_name=False)
            swap.click(
                lambda w, h: (h, w), [width, height], [width, height],
                queue=False, api_name="swap_dimensions",
            )
        with gr.Accordion("モデル・保存先", open=False):
            gr.Markdown(
                "生成した PNG と設定 JSON は `outputs/` に保存されます。\n\n"
                "モデル: [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)\n\n"
                "編集指示の補助: "
                "[Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)"
            )
            unload = gr.Button("モデルを解放", elem_classes="wb-action")
            status = gr.Textbox(label="状態", interactive=False)
            unload.click(manager.unload, outputs=status, concurrency_id="inference")

        editable = [task, prompt, uploader, upload_mode, thorough, width, height, steps,
                    seed, transparent, swap, unload]
        control_outputs = [*editable, run, enhance, combined, clear_references, reuse_result, action_hint,
                           example_controls]
        control_inputs = [busy, active_operation, task, references, prompt, output]

        def controls(is_busy, operation, mode, paths, instruction, result):
            allowed = action_availability(mode, paths, instruction, result, is_busy)
            updates = {component: gr.update(interactive=allowed["edit"]) for component in editable}
            updates.update({
                run: gr.update(interactive=allowed["generate"], value=(
                    "画像を生成中…" if is_busy and operation == "generate" else "画像を生成"
                )),
                enhance: gr.update(interactive=allowed["enhance"], value=(
                    "指示を整理中…" if is_busy and operation == "enhance" else "編集指示を整える"
                )),
                combined: gr.update(interactive=allowed["combined"], value=(
                    "指示整理 → 生成中…" if is_busy and operation == "combined" else "指示を整えてから生成"
                )),
                clear_references: gr.update(interactive=allowed["clear"]),
                reuse_result: gr.update(interactive=allowed["reuse"]),
                action_hint: allowed["reason"],
                example_controls: gr.update(visible=not is_busy),
            })
            return updates

        def begin(operation, is_busy, mode, paths, instruction, result):
            allowed = action_availability(mode, paths, instruction, result, is_busy)
            if not allowed[operation]:
                raise gr.Error(allowed["reason"])
            return {
                **controls(True, operation, mode, paths, instruction, result),
                busy: True, active_operation: operation,
                run_status: status_card("queued", "処理の開始を待っています。", operation),
            }

        def finish(mode, paths, instruction, result):
            return {
                **controls(False, "", mode, paths, instruction, result),
                busy: False, active_operation: "",
            }

        def refresh(is_busy, operation, mode, paths, instruction, result):
            # Streamed output changes must not race the explicit finish transition.
            if is_busy:
                return {component: gr.skip() for component in control_outputs}
            return controls(False, operation, mode, paths, instruction, result)

        gr.on(
            [task.change, references.change, prompt.change],
            refresh, control_inputs, control_outputs, queue=False,
            show_progress="hidden", api_name=False,
        )
        for button, operation, function, inputs, outputs, api_name in [
            (run, "generate", stream_generate_image,
             [task, references, prompt, width, height, steps, seed, transparent],
             [output, files, info, run_status, generation_time, run_log], "generate_image"),
            (combined, "combined", stream_enhance_and_generate,
             [task, references, prompt, width, height, steps, seed, transparent, thorough],
             [prompt, output, files, info, run_status, generation_time, enhancement_time, run_log],
             "enhance_and_generate"),
            (enhance, "enhance", stream_enhance_prompt,
             [task, references, prompt, thorough],
             [prompt, run_status, enhancement_time, run_log], "enhance_prompt"),
        ]:
            def start(is_busy, mode, paths, instruction, result, operation=operation):
                return begin(operation, is_busy, mode, paths, instruction, result)

            execution = button.click(
                start, [busy, task, references, prompt, output],
                [busy, active_operation, run_status, *control_outputs],
                queue=False, api_name=False,
            ).success(
                function, inputs, outputs, concurrency_limit=1, concurrency_id="inference",
                show_progress="hidden", api_name=api_name,
            )
            for completion in (execution.success, execution.failure):
                completion(
                    finish, [task, references, prompt, output],
                    [busy, active_operation, *control_outputs], queue=False, api_name=False,
                )
    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local image workbench")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    configure_logging()
    build_app().launch(
        server_name=args.host, server_port=args.port, share=False,
        theme=gr.themes.Soft(primary_hue="teal"),
        css=UI_CSS, js=UI_JS,
    )
