from __future__ import annotations

import json
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Iterator, TypedDict

from openai import OpenAI

from app.context import Context
from app.events import AssistantEvent, ErrorEvent, TranscriptionEvent
from app.openai import chat_with_json_schema


@dataclass(frozen=True)
class ChatConfig:
    model: str = "gpt-5-chat-latest"


class AssistantPayload(TypedDict):
    message: str


class ChatWorker:
    def __init__(self, client: OpenAI, config: ChatConfig) -> None:
        self._client = client
        self._config = config

    def ask(self, context: Context) -> AssistantPayload:
        payload = chat_with_json_schema(
            self._client,
            self._config.model,
            context.to_openai_messages(),
            {
                "name": "assistant_message",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        )
        message = payload.get("message")
        if isinstance(message, str):
            return {"message": message.strip()}
        return {"message": json.dumps(payload, ensure_ascii=False)}


def chat_worker(
    input_queue: Queue[object],
    stop_event,
    worker: ChatWorker,
    context: Context,
) -> Iterator[AssistantEvent | ErrorEvent | None]:
    # 分割書き起こしを受け取り、チャット応答イベントを発行する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, TranscriptionEvent):
            yield None
            continue
        text = event.text.strip()
        if not text:
            yield None
            continue
        context.add_user_message(text)
        try:
            payload = worker.ask(context)
        except Exception as exc:
            yield ErrorEvent(topic="assistant_error", message=str(exc))
            continue
        message = payload.get("message", "").strip()
        if message:
            context.add_assistant_message(message)
            yield AssistantEvent(topic="assistant_output", payload=payload)
