from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock
from typing import Iterable


@dataclass(frozen=True)
class AudioChunkEvent:
    topic: str
    chunk: object


@dataclass(frozen=True)
class TranscriptionEvent:
    topic: str
    kind: str
    text: str
    transcribe_sec: float
    chunk_sec: float
    chunk_index: int


@dataclass(frozen=True)
class AssistantEvent:
    topic: str
    payload: dict[str, object]


@dataclass(frozen=True)
class ErrorEvent:
    topic: str
    message: str


@dataclass(frozen=True)
class ScheduleFireEvent:
    topic: str
    fired_at: float
    schedule_type: str


@dataclass(frozen=True)
class SystemEvent:
    topic: str
    message: str


class EventBroker:
    def __init__(self, stop_event: Event) -> None:
        self._stop_event = stop_event
        self._subscriptions: dict[str, list[Queue[object]]] = {}
        self._lock = Lock()

    def publish(self, event: object) -> None:
        topic = getattr(event, "topic", None)
        if not isinstance(topic, str):
            raise ValueError("Event must have topic: str")
        for queue in self._get_subscribers(topic):
            queue.put(event)

    def subscribe(self, topic: str) -> Queue[object]:
        with self._lock:
            queue: Queue[object] = Queue()
            self._subscriptions.setdefault(topic, []).append(queue)
            return queue

    def queue_sizes(self) -> list[tuple[str, int]]:
        sizes: list[tuple[str, int]] = []
        with self._lock:
            for topic, queues in self._subscriptions.items():
                for queue in queues:
                    sizes.append((topic, queue.qsize()))
        return sizes

    def stop_event(self) -> Event:
        return self._stop_event

    def _get_subscribers(self, topic: str) -> Iterable[Queue[object]]:
        with self._lock:
            return list(self._subscriptions.get(topic, []))
