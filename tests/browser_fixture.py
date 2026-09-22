"""Deterministic inference fixture for the browser interaction test."""

import json
import sys
import time
from pathlib import Path
from PIL import Image
import app
from prompt_enhancer import IncompleteRewrite

root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
for color in ["red", "blue", "green"]:
    Image.new("RGB", (384, 256), color).save(root / f"{color}.png")


def generate(paths, options, progress):
    with (root / "generate-calls.txt").open("a") as calls:
        calls.write(options.prompt + "\n")
    progress(0, desc="モデルを読み込み中")
    time.sleep(0.5)
    for step in range(80):
        progress((step + 1) / 80, desc=f"生成中 {step + 1}/80 ステップ")
        time.sleep(0.055)
    if "FAIL" in options.prompt:
        raise RuntimeError("Test generation failure")
    metadata = {
        "prompt": options.prompt,
        "width": options.width,
        "height": options.height,
        "load_seconds": 0.5,
        "inference_seconds": 4.4,
        "reference_count": len(paths),
        "references": [Path(p).name for p in paths],
    }
    img = root / "result.png"
    Image.new("RGB", (options.width, options.height), "purple").save(img)
    data = root / "result.json"
    data.write_text(json.dumps(metadata))
    return str(img), str(data), metadata


def rewrite(paths, prompt, progress, thorough=False):
    for i in range(20):
        progress(0.5, desc=f"指示を整理中 {i + 1} トークン")
        time.sleep(0.08)
    if "INCOMPLETE" in prompt:
        raise IncompleteRewrite("Test incomplete answer")
    app.manager.rewrite_stats = {"load_seconds": 0.4, "generate_seconds": 1.2, "cache_hit": False}
    return "Rewritten: " + prompt


app.manager.generate = generate
app.manager.rewrite_prompt = rewrite
app.build_app().launch(
    server_name="127.0.0.1",
    server_port=int(sys.argv[2]),
    share=False,
    theme=app.gr.themes.Soft(primary_hue="teal"),
    allowed_paths=[str(root)],
    css=app.UI_CSS,
    js=app.UI_JS,
)
