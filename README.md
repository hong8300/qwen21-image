# Image Workbench

Qwen Image 2.1 の画像編集・画像生成を利用する日本語 Gradio アプリです。推論はローカルで実行します。

## 起動

[uv](https://docs.astral.sh/uv/) と Python 3.12 を使用します。

```bash
uv sync
uv run python app.py
```

ブラウザで http://127.0.0.1:7860 を開きます。モデルは初回実行時に Hugging Face からダウンロードされ、通常は `~/.cache/huggingface/hub/` にキャッシュされます。モデルは約 33 GB です。API キーは不要です。

同じ LAN の別マシンからは `http://<起動したマシンのIPアドレス>:7860` を開きます。
既定では `0.0.0.0` で待ち受けます。待ち受け設定の変更を反映するには、起動中のアプリを Ctrl+C で停止して再起動してください。

## 使い方

- **Image to Image**: 参照画像を 1〜10 枚アップロードし、編集指示を入力して「画像を生成」。参照番号はアップロード順です。
- **Text to Image**: モードを切り替えて文章から画像を生成。
- 透明背景のチェックを入れると、公式推奨の RGBA 指示をプロンプトに追加します。透過結果はモデルの出力に依存します。保存時にアルファチャンネルを保持します。
- シード `-1` はランダム。生成情報に実際のシードを記録します。同じ結果の再現にはプロンプト・画像・解像度・ステップ数・環境も揃えてください。
- PNG と生成設定 JSON は `outputs/` に保存され、画面からダウンロードできます。

画像生成・編集ともに `Qwen/Qwen-Image-2.1` を使用します。

## 実行環境

CUDA → Apple Silicon MPS → CPU の順で自動選択します。生成は既定で 1024×1024 / 40 steps。CPU 推論は非常に時間がかかります。Apple Silicon 対応は MPS を利用し、未対応の演算は PyTorch の CPU フォールバックを有効にしています。モデルのロードと推論には十分な空きメモリが必要です。

CUDA では既定でモデル CPU オフロードを有効にします。MPS はメモリ上にモデルを配置します。一度読み込んだモデルを生成・編集の両方で再利用します。同時推論は 1 件に制限しています。「モデルを解放」でメモリを解放できます。

| 環境変数 | 既定値 | 用途 |
| --- | --- | --- |
| `QWEN_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `QWEN_CPU_OFFLOAD` | `1` | CUDA で CPU オフロード。`0` で無効 |
| `OUTPUT_DIR` | プロジェクト内 `outputs/` | PNG・設定 JSON の保存先 |
| `HF_HOME` | Hugging Face の既定値 | モデルのキャッシュ保存先 |

```bash
QWEN_DEVICE=mps uv run python app.py --port 7861
```

このマシンからだけ利用する場合は `uv run python app.py --host 127.0.0.1` で起動します。

## 開発・検証

```bash
uv run pytest -q
uv run ruff check .
```

単体テストは大きなモデルを読み込まず、入力検証・参照画像順序・シード・透過 PNG 保存・推論への引数を確認します。実モデルの品質・速度の検証は別途推論が必要です。

学習済みモデルで生成→編集をまとめて確認する場合:

```bash
uv run python smoke.py --size 512 --steps 2
```

2 steps は動作確認用です。画質を確認する場合は `--steps 40` を指定してください。

2026-09-21 に Apple Silicon / メモリ 128 GB / MPS で確認済み:

- 単体テスト 12 件、Ruff チェック成功。
- 実モデルで 512×512 / 40 steps / seed 42 の Text to Image が約 13.5 秒（ロード後）。
- 生成した赤いティーポットを青に変える Image to Image が約 22.4 秒。
- Gradio 画面で生成結果・PNG ダウンロードを確認。編集は Gradio API 経由で確認。

上記はこの環境での測定値です。解像度・参照画像数・ハードウェアによって変わります。

Diffusers は Qwen Image 2.1 対応済みコミットに固定しています。依存バージョンは `uv.lock` で固定します。

## 公式資料

- [Qwen Image 2.1 / 推論例・モデルライセンス](https://github.com/QwenLM/Qwen-Image-2.1)
- [Qwen Image 2.1 モデル](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Diffusers Qwen Image 2.1 実装](https://github.com/huggingface/diffusers/tree/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21)
