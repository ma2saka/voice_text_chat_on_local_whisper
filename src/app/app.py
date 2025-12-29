from __future__ import annotations

import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

from openai import OpenAI

from app.bootstrap import bootstrap
from app.config import AudioChunkConfig, WhisperConfig, WorkerConfig
from app.context import Context
from app.events import EventBroker, ScheduleFireEvent
from app.worker import run_worker
from app.workers.chat import ChatConfig, ChatWorker, chat_worker
from app.workers.cron import cron_worker
from app.workers.display import (
    assistant_display_worker,
    realtime_display_worker,
    system_display_worker,
    transcribe_error_display_worker,
    user_display_worker,
)
from app.workers.listner import (
    ChunkProcessor,
    MicrophoneEventSource,
    listener_worker,
)
from app.workers.monitor import monitor_worker
from app.workers.think import ThinkConfig, ThinkWorker, think_worker
from app.workers.transcribe import TranscribeWorker, transcribe_worker


def build_system_prompt() -> str:
    return (
        "あなたは日本語で会話するアシスタントスタッフです。\n"
        f"現在日付は{datetime.now()}です\n"
        "ユーザーをサポートするアシスタントサービスをやってるアルバイトスタッフとして会話に応じます。\n"
        "チャットが始まったときは少し事務的ですが、少しやりとりをするとくだけた口調になります。\n"
        "あなたは優秀でユーモアのある人物です。ユーザーの話をよく聞き、適切に反応します。\n"
        "チャットが始まったときはお互いに相手のことを知りません。アルバイトのルールで、名前など個人情報を話してはいけない決まりになっていますが、個人が特定されない範囲であればいいだろう、と思っています。\n"
        "質問して答えを引き出すのではなく、自分なりの意見や設定に基づく自己開示で会話をリードします。"
    )


def build_think_prompt(thinking: str | None) -> str:
    normalized_thinking = thinking or "なし"
    return (
        "会話履歴と現在のthinkingを踏まえて、"
        "アシスタントの思考背景と設定を更新してください。十分な情報がなければ創作してよいです。"
        "これまでの会話についてどう考えているのか。ユーザーのことはどういう人物だと考えているのか。"
        "そういうことも整理します。一貫性のあるキャラクター設定と会話を行うための重要な情報です。"
        "必ず「アシスタントが今考えていることと今の状態:」から始めてください。"
        f"\n\n現在のthinking: {normalized_thinking}"
    )


def main() -> None:
    bootstrap()

    logger = _build_logger()
    context = Context(system_prompt=build_system_prompt())
    client = OpenAI()
    audio_config = AudioChunkConfig()
    whisper_config = WhisperConfig()
    stop_event = Event()
    broker = EventBroker(stop_event)
    mic_source = MicrophoneEventSource(audio_config, stop_event)
    processor = ChunkProcessor(audio_config)
    transcriber = TranscribeWorker(whisper_config)
    chat_agent = ChatWorker(client, ChatConfig())
    thinker = ThinkWorker(client, ThinkConfig(), build_think_prompt)
    audio_chunk_queue = broker.subscribe("audio_chunk")
    split_transcription_for_chat = broker.subscribe("split_transcription")
    split_transcription_for_display = broker.subscribe("split_transcription")
    realtime_transcription_queue = broker.subscribe("realtime_transcription")
    assistant_output_queue = broker.subscribe("assistant_output")
    assistant_error_queue = broker.subscribe("assistant_error")
    transcribe_error_queue = broker.subscribe("transcribe_error")
    schedule_fire_queue_for_think = broker.subscribe("schedule_fire")
    system_output_queue = broker.subscribe("system_output")
    think_error_queue = broker.subscribe("think_error")
    # ワーカー一覧（入力→処理→publish の流れ）
    worker_threads = [
        # マイク入力をチャンク化
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=mic_source,
                    stop_event=stop_event,
                    worker_fn=listener_worker,
                    publisher=broker,
                    args=(processor,),
                ),
            ),
            daemon=True,
        ),
        # チャンクの書き起こし
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=audio_chunk_queue,
                    stop_event=stop_event,
                    worker_fn=transcribe_worker,
                    publisher=broker,
                    args=(transcriber, audio_config, whisper_config, logger),
                ),
            ),
            daemon=True,
        ),
        # 分割書き起こしからチャット応答
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=split_transcription_for_chat,
                    stop_event=stop_event,
                    worker_fn=chat_worker,
                    publisher=broker,
                    args=(chat_agent, context),
                ),
            ),
            daemon=True,
        ),
        # リアルタイム書き起こし表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=realtime_transcription_queue,
                    stop_event=stop_event,
                    worker_fn=realtime_display_worker,
                    publisher=None,
                    args=(2.0,),
                ),
            ),
            daemon=True,
        ),
        # ユーザー発話表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=split_transcription_for_display,
                    stop_event=stop_event,
                    worker_fn=user_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
        # アシスタント応答表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=assistant_output_queue,
                    stop_event=stop_event,
                    worker_fn=assistant_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
        # アシスタントエラー表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=assistant_error_queue,
                    stop_event=stop_event,
                    worker_fn=assistant_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
        # 書き起こしエラー表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=transcribe_error_queue,
                    stop_event=stop_event,
                    worker_fn=transcribe_error_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
        # thinkingエラー表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=broker,
                    stop_event=stop_event,
                    worker_fn=monitor_worker,
                    publisher=None,
                    args=(logger, 5.0),
                ),
            ),
            daemon=True,
        ),
        # キュー監視ログ
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=None,
                    stop_event=stop_event,
                    worker_fn=cron_worker,
                    publisher=broker,
                    args=(15.0, 60.0),
                ),
            ),
            daemon=True,
        ),
        # schedule_fire を定期発行
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=schedule_fire_queue_for_think,
                    stop_event=stop_event,
                    worker_fn=think_worker,
                    publisher=broker,
                    args=(thinker, context, logger),
                ),
            ),
            daemon=True,
        ),
        # thinking 更新
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=system_output_queue,
                    stop_event=stop_event,
                    worker_fn=system_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
        # システム表示
        Thread(
            target=run_worker,
            args=(
                WorkerConfig(
                    event_source=think_error_queue,
                    stop_event=stop_event,
                    worker_fn=transcribe_error_display_worker,
                    publisher=None,
                ),
            ),
            daemon=True,
        ),
    ]
    for thread in worker_threads:
        thread.start()
    broker.publish(
        ScheduleFireEvent(
            topic="schedule_fire",
            fired_at=time.time(),
            schedule_type="think_update",
        )
    )
    print(
        "## アシスタントと接続できました。どうぞ対話を楽しんでください。 ##", flush=True
    )
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for thread in worker_threads:
            thread.join(timeout=1.0)


if __name__ == "__main__":
    main()


def _build_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    log_path = log_dir / f"app-{timestamp}.log"
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
