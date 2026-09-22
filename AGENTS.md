# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## リポジトリの内容

Image Workbench for Mac。Apple SiliconのMPSでQwen Image 2.1のローカル推論を行うGradioアプリ。

## 実行方法

Python 3.12とuvを使用する。

```bash
uv sync
uv run python app.py --host 127.0.0.1
uv run pytest -q
uv run pytest tests/test_execution.py -q
uv run ruff check .
uv run --with playwright python tests/browser_ui.py
```

## 規約・地雷

- ブラウザ検証は一時プロファイルのheadless Chromiumを使う。ユーザーの通常のChromeでタブを開くと他のマシンへ同期されるため使用しない。
- 起動中のユーザー用サーバーを止めず、検証用サーバーは別ポートで立てて終了時に停止する。
- 単体・ブラウザ回帰テストの推論は代替処理。実モデルの画質や速度を確認したことにはならない。
- Gradioのストリーミング失敗後は汎用のthenだけでは操作の復帰が不安定だった。successとfailureの両経路で復帰を検証する。
- 指示欄の変更検知はchangeを使う。inputでは貼り付け等を拾えず、実行ボタンが有効にならない事象があった。
