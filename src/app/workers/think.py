from __future__ import annotations

import json
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Iterator

from openai import OpenAI

from app.context import Context
from app.events import ErrorEvent, ScheduleFireEvent, SystemEvent
from app.openai import chat_text


@dataclass(frozen=True)
class ThinkConfig:
    model: str = "gpt-5.2"


class ThinkWorker:
    def __init__(
        self,
        client: OpenAI,
        config: ThinkConfig,
        prompt_builder: Callable[[str | None], str],
    ) -> None:
        self._client = client
        self._config = config
        self._prompt_builder = prompt_builder

    def summarize(self, context: Context) -> str:
        messages = context.to_openai_messages()
        messages.append(
            {
                "role": "user",
                "content": self._prompt_builder(context.thinking),
            }
        )
        return chat_text(
            self._client,
            self._config.model,
            messages,
        )


def think_worker(
    input_queue: Queue[object],
    stop_event,
    worker: ThinkWorker,
    context: Context,
    logger,
) -> Iterator[SystemEvent | ErrorEvent | None]:
    # schedule_fire を受けて thinking を更新し、通知イベントを発行する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, ScheduleFireEvent):
            yield None
            continue
        if event.schedule_type != "think_update":
            yield None
            continue
        _drain_queue(input_queue)
        try:
            summary = worker.summarize(context)
        except Exception as exc:
            yield ErrorEvent(topic="think_error", message=str(exc))
            continue
        context.set_thinking(summary)
        logger.info(
            json.dumps(
                {"event": "thinking_update", "thinking": summary},
                ensure_ascii=False,
            )
        )
        yield SystemEvent(
            topic="system_output",
            message="  * アシスタントの現状認識がアップデートされました *",
        )


def _drain_queue(queue: Queue[object]) -> None:
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break
