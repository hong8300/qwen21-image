"""Browser regression test using an isolated, headless browser and fake inference.

Run: uv run --with playwright python tests/browser_ui.py
"""

import base64
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check_ui(url, root):
    expect.set_options(timeout=15000)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.set_default_timeout(15000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        run = page.locator("#generate-button")
        prompt = page.locator("#instruction textarea")
        expect(run).to_be_disabled()
        disabled_color = run.evaluate("(e)=>getComputedStyle(e).backgroundColor")

        def drop(name):
            payload = {
                "data": base64.b64encode((root / f"{name}.png").read_bytes()).decode(),
                "name": name + ".png",
            }
            dt = page.evaluate_handle(
                """p=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob(p.data),c=>c.charCodeAt(0))],p.name,{type:'image/png'}));return d;}""",
                payload,
            )
            zone = page.locator("#reference-upload .upload-container")
            zone.dispatch_event("dragover", {"dataTransfer": dt})
            zone.dispatch_event("drop", {"dataTransfer": dt})
            page.wait_for_function(
                '(name)=>Array.from(document.querySelectorAll("#reference-preview img")).some(e=>e.src.includes(name))',
                arg=name + ".png",
            )
            expect(zone).to_be_visible()

        drop("red")
        expect(page.locator("#reference-count")).to_contain_text("1 / 10枚")
        drop("blue")
        expect(page.locator("#reference-count")).to_contain_text("1 / 10枚")
        page.get_by_role("radio", name="追加", exact=True).check()
        drop("green")
        expect(page.locator("#reference-count")).to_contain_text("2 / 10枚")
        prompt.fill("Combine the reference images")
        expect(run).to_be_enabled()
        page.wait_for_function(
            "()=>getComputedStyle(document.querySelector('#generate-button')).backgroundColor==='rgb(15, 118, 110)'"
        )
        enabled_color = run.evaluate("(e)=>getComputedStyle(e).backgroundColor")
        assert disabled_color != enabled_color, (disabled_color, enabled_color)
        page.get_by_role("button", name=re.compile("^実行ログ")).click()
        run.click()
        expect(page.locator('#run-status [data-state="running"]')).to_be_visible()
        expect(run).to_be_disabled()
        expect(run).to_contain_text("生成中")
        expect(prompt).not_to_be_editable()
        expect(page.locator("#clear-references")).to_be_disabled()
        expect(page.locator("#reuse-result")).to_be_disabled()
        page.wait_for_function(
            "()=>getComputedStyle(document.querySelector('#generate-button')).backgroundColor==='rgb(254, 243, 199)'"
        )
        busy_color = run.evaluate("(e)=>getComputedStyle(e).backgroundColor")
        assert busy_color != disabled_color
        page.wait_for_function(
            """()=>{const e=document.querySelector('#run-log textarea');return e && e.scrollHeight>e.clientHeight && e.scrollTop+e.clientHeight>=e.scrollHeight-3;}"""
        )
        page.screenshot(path=str(root / "running.png"), full_page=True)
        # Reading older lines with follow disabled must work while new logs arrive.
        page.locator("#follow-log input").uncheck()
        before = page.locator("#run-log textarea").input_value()
        page.locator("#run-log textarea").evaluate("(e)=>e.scrollTop=0")
        page.wait_for_function(
            "before => document.querySelector('#run-log textarea').value !== before", arg=before
        )
        assert page.locator("#run-log textarea").evaluate("(e)=>e.scrollTop") == 0
        page.locator("#follow-log input").check()
        page.wait_for_function(
            "()=>{const e=document.querySelector('#run-log textarea');return e.scrollTop+e.clientHeight>=e.scrollHeight-3;}"
        )
        expect(page.locator('#run-status [data-state="success"]')).to_be_visible()
        expect(run).to_be_enabled()
        expect(prompt).to_be_editable()
        expect(page.locator("#reuse-result")).to_be_enabled()
        # Reopen after completion: the newest line must be visible.
        page.get_by_role("button", name=re.compile("^実行ログ")).click()
        page.get_by_role("button", name=re.compile("^実行ログ")).click()
        page.wait_for_function(
            """()=>{const e=document.querySelector('#run-log textarea');return e.scrollTop+e.clientHeight>=e.scrollHeight-3;}"""
        )
        # Turn follow off and read old lines without being pulled back down.
        page.locator("#follow-log input").uncheck()
        page.locator("#run-log textarea").evaluate("(e)=>e.scrollTop=0")
        page.wait_for_timeout(400)
        assert page.locator("#run-log textarea").evaluate("(e)=>e.scrollTop") == 0
        page.locator("#latest-log").click()
        page.wait_for_function(
            """()=>{const e=document.querySelector('#run-log textarea');return e.scrollTop+e.clientHeight>=e.scrollHeight-3;}"""
        )
        # A generated output becomes the next editing reference.
        page.locator("#reuse-result").click()
        expect(page.locator("#reference-count")).to_contain_text("1 / 10枚")
        page.wait_for_function(
            '()=>Array.from(document.querySelectorAll("#reference-preview img")).some(e=>e.src.includes("result.png"))'
        )
        page.get_by_role("radio", name="入れ替え", exact=True).check()
        drop("red")
        prompt.fill("Second edit")
        run.click()
        expect(page.locator('#run-status [data-state="running"]')).to_be_visible()
        expect(page.locator('#run-status [data-state="success"]')).to_be_visible()
        expect(run).to_be_enabled()
        # A failed request must also restore controls.
        prompt.fill("FAIL")
        run.click()
        expect(page.locator('#run-status [data-state="error"]')).to_be_visible()
        expect(run).to_be_enabled()
        expect(prompt).to_be_editable()
        # The incomplete rewrite path must preserve the input and restore controls.
        prompt.fill("INCOMPLETE")
        page.locator("#enhance-button").click()
        expect(page.locator('#run-status [data-state="warning"]')).to_be_visible()
        expect(page.locator("#enhance-button")).to_be_enabled()
        expect(prompt).to_have_value("INCOMPLETE")
        page.locator("#clear-references").click()
        expect(page.locator("#reference-count")).to_contain_text("0 / 10枚")
        expect(run).to_be_disabled()
        page.get_by_role("radio", name="Text to Image", exact=True).check()
        expect(page.locator("#reference-upload")).not_to_be_visible()
        expect(run).to_be_enabled()
        page.get_by_role("radio", name="Image to Image", exact=True).check()
        expect(run).to_be_disabled()
        drop("blue")
        expect(run).to_be_enabled()
        prompt.fill("Final edit")
        page.locator("#enhance-button").click()
        expect(page.locator('#run-status [data-state="success"]')).to_be_visible()
        expect(page.locator("#enhance-button")).to_be_enabled()
        expect(prompt).to_have_value("Rewritten: Final edit")
        page.screenshot(path=str(root / "complete.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
        page.screenshot(path=str(root / "mobile.png"), full_page=True)
        assert not errors, errors
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "disabled_color": disabled_color,
                    "enabled_color": enabled_color,
                    "busy_color": busy_color,
                    "browser_errors": errors,
                },
                ensure_ascii=False,
            )
        )
        browser.close()


def main():
    project = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="qwen-ui-test-") as directory:
        root = Path(directory)
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(project / "tests/browser_fixture.py"), str(root), str(port)],
                cwd=project,
                env={**os.environ, "PYTHONPATH": str(project)},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 30
                while True:
                    try:
                        with urllib.request.urlopen(url + "/config", timeout=1):
                            break
                    except OSError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError((root / "server.log").read_text()) from None
                        time.sleep(0.1)
                check_ui(url, root)
                artifact_dir = os.getenv("UI_TEST_ARTIFACTS")
                if artifact_dir:
                    import shutil

                    target = Path(artifact_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    for image in root.glob("*.png"):
                        if image.name in {"running.png", "complete.png", "mobile.png"}:
                            shutil.copy2(image, target / image.name)
            finally:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
