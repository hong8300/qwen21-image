"""UI states and browser behavior for repeatable image editing workflows."""

from html import escape


def status_card(state, message, operation=""):
    labels = {
        "idle": "待機中", "queued": "開始待ち", "running": "現在処理中",
        "success": "完了", "warning": "未完了", "error": "エラー",
    }
    label = labels[state]
    if state == "running":
        label = "現在生成中" if operation.endswith("generate") else "現在指示を整理中"
    return (
        f'<div class="status-card" data-state="{state}" data-operation="{operation}" '
        f'role="status" aria-live="polite" aria-atomic="true">'
        f'<strong><span class="status-dot" aria-hidden="true"></span>{label}</strong>'
        f'<span class="status-detail">{escape(message)}</span></div>'
    )


def action_availability(mode, paths, prompt, output, busy=False):
    references_ok = 1 <= len(paths or []) <= 10
    prompt_ok = bool((prompt or "").strip())
    image_mode = mode == "Image to Image"
    if busy:
        reason = "処理中です。完了すると、参照画像や指示を変更できます。"
    elif image_mode and not references_ok:
        reason = "参照画像を1〜10枚ドロップしてください。生成結果も参照画像に使えます。"
    elif not prompt_ok:
        reason = "編集・生成の指示を入力すると、実行ボタンが有効になります。"
    else:
        reason = "準備できました。青緑のボタンから実行できます。"
    return {
        "generate": not busy and prompt_ok and (not image_mode or references_ok),
        "combined": not busy and prompt_ok and image_mode and references_ok,
        "enhance": not busy and prompt_ok and image_mode and references_ok,
        "clear": not busy and bool(paths),
        "reuse": not busy and bool(output),
        "edit": not busy,
        "reason": reason,
    }


def combine_references(uploaded, current, mode):
    if not uploaded:
        return list(current or [])
    paths = list(current or []) + list(uploaded) if mode == "追加" else list(uploaded)
    if len(paths) > 10:
        raise ValueError("参照画像は最大10枚です。入れ替えるか、枚数を減らしてください。")
    return paths


UI_CSS = """
.status-card {
  display: flex; flex-direction: column; gap: 4px; padding: 10px 14px;
  border: 2px solid #94a3b8; border-radius: 12px; background: #f1f5f9; color: #334155;
}
.status-card strong { display: flex; align-items: center; gap: 10px; font-size: 18px; }
.status-detail { line-height: 1.6; overflow-wrap: anywhere; }
.status-dot { width: 12px; height: 12px; border-radius: 50%; background: currentColor; }
.status-card[data-state="queued"], .status-card[data-state="running"] {
  color: #854d0e; background: #fef3c7; border-color: #d97706;
}
.status-card[data-state="running"] .status-dot { animation: workbench-pulse 1.4s infinite; }
.status-card[data-state="success"] { color: #166534; background: #dcfce7; border-color: #16a34a; }
.status-card[data-state="warning"] { color: #9a3412; background: #ffedd5; border-color: #ea580c; }
.status-card[data-state="error"] { color: #991b1b; background: #fee2e2; border-color: #dc2626; }
button.wb-action { min-height: 44px; font-weight: 650; border: 1px solid #0f766e;
  transition: none !important; }
button.wb-action:not(:disabled) { color: white; background: #0f766e; cursor: pointer; }
button.wb-action:not(:disabled):hover { background: #115e59; }
button.wb-action:disabled {
  color: #475569 !important; background: #e2e8f0 !important; border-color: #cbd5e1 !important;
  opacity: 1 !important; cursor: not-allowed;
}
button.wb-action:focus-visible { outline: 3px solid #0ea5e9; outline-offset: 3px; }
body:has(.status-card[data-operation="generate"][data-state="running"]) #generate-button,
body:has(.status-card[data-operation="enhance"][data-state="running"]) #enhance-button,
body:has(.status-card[data-operation^="combined-"][data-state="running"]) #combined-button {
  color: #854d0e !important; background: #fef3c7 !important; border-color: #d97706 !important;
}
button.wb-action * { color: inherit !important; }
#run-status .status-card * { color: inherit !important; }
.gradio-container { max-width: 1440px !important; --layout-gap: 10px; }
#workspace, #workspace .gr-form, #workspace .gap { gap: 10px; }
#app-heading h1 { margin-bottom: 2px; font-size: 24px; }
#app-heading p { margin: 0; }
#action-hint p, #reference-count p { margin: 0; font-size: 13px; }
#reference-upload { border: 2px dashed #0f766e; border-radius: 12px; }
#reference-count { margin-top: 4px; }
#run-log textarea { height: 280px !important; max-height: 280px !important;
  overflow-y: auto !important; font-family: ui-monospace, monospace; font-size: 12px; }
@keyframes workbench-pulse { 50% { opacity: 0.35; } }
@media (prefers-reduced-motion: reduce) { .status-dot { animation: none !important; } }
"""

# Textarea values are DOM properties, so a MutationObserver alone misses updates.
UI_JS = """() => {
  if (window.workbenchLogTimer) clearInterval(window.workbenchLogTimer);
  let previous = {value: null, visible: false, height: 0, follow: false};
  window.workbenchLogTimer = setInterval(() => {
    const area = document.querySelector('#run-log textarea');
    const check = document.querySelector('#follow-log input[type="checkbox"]');
    if (!area || !check) return;
    const visible = area.getClientRects().length > 0 && area.clientHeight > 0;
    const changed = area.value !== previous.value || visible !== previous.visible
      || area.scrollHeight !== previous.height || check.checked !== previous.follow;
    if (check.checked && visible && changed) area.scrollTop = area.scrollHeight;
    previous = {value: area.value, visible, height: area.scrollHeight, follow: check.checked};
  }, 200);
  window.addEventListener('pagehide', () => clearInterval(window.workbenchLogTimer), {once: true});
}"""

SCROLL_TO_LATEST = """() => {
  const area = document.querySelector('#run-log textarea');
  if (area) area.scrollTop = area.scrollHeight;
}"""
