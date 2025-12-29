# voice_text_chat_on_local_whisper

マイク入力をリアルタイムで書き起こしし、分割チャンクではOpenAI応答まで行う実験用プロジェクトです。NVidia のGPUがある環境を想定しています。
イベント駆動（ブロードキャスト）でワーカーを分離しています。

## Quick Start
```
uv run -m app
```

## 必要なもの
- Python 3.13
- `.env` に `OPENAI_API_KEY`
- GPU利用時: `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`
- 触れるマイク入力

## ざっくり構成
- `src/app/app.py`: 起動・ワーカー配線
- `src/app/workers/*`: 各ワーカー
- `src/app/events.py`: イベントブローカー
- `src/app/context.py`: 会話履歴・thinking
- `src/app/bootstrap.py`: dotenv + CUDA準備
