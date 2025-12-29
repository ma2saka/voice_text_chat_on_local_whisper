from __future__ import annotations

import json
import sys
import time
from queue import Empty, Queue
from typing import Iterator

from app.events import AssistantEvent, ErrorEvent, SystemEvent, TranscriptionEvent


def realtime_display_worker(
    input_queue: Queue[object],
    stop_event,
    stale_seconds: float = 2.0,
) -> Iterator[None]:
    # リアルタイム書き起こしの最終行を更新し、無更新なら消去する。
    last_update = 0.0
    has_status = False
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            if has_status and (time.perf_counter() - last_update) >= stale_seconds:
                _clear_status_line()
                has_status = False
            yield None
            continue
        if not isinstance(event, TranscriptionEvent):
            yield None
            continue
        message = f"聞き取っています: {event.text}"
        _write_status_line(message)
        last_update = time.perf_counter()
        has_status = True
        yield None


def user_display_worker(
    input_queue: Queue[object],
    stop_event,
) -> Iterator[None]:
    # 分割書き起こしをユーザー発話として表示する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, TranscriptionEvent):
            yield None
            continue
        _clear_status_line()
        sys.stdout.write(
            "\033[96mUser:\033[0m "
            f"{event.text} "
            "\033[90m"
            f"(transcribe: {event.transcribe_sec:.2f} sec, "
            f"chunk {event.chunk_sec:.2f} sec)"
            "\033[0m\n"
        )
        sys.stdout.flush()
        yield None


def assistant_display_worker(
    input_queue: Queue[object],
    stop_event,
) -> Iterator[None]:
    # アシスタント応答やエラーを表示する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, (AssistantEvent, ErrorEvent)):
            yield None
            continue
        _clear_status_line()
        if isinstance(event, AssistantEvent):
            sys.stdout.write(
                "\033[92mAssistant:\033[0m "
                f"{event.payload.get('message', '')}\n"
            )
        else:
            sys.stdout.write(f"Assistant error: {event.message}\n")
        sys.stdout.flush()
        yield None


def system_display_worker(
    input_queue: Queue[object],
    stop_event,
) -> Iterator[None]:
    # システムメッセージの表示を担当する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, SystemEvent):
            yield None
            continue
        _clear_status_line()
        sys.stdout.write(f"\033[93mSystem:\033[0m {event.message}\n")
        sys.stdout.flush()
        yield None


def transcribe_error_display_worker(
    input_queue: Queue[object],
    stop_event,
) -> Iterator[None]:
    # 書き起こし/thinkのエラー表示を担当する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        if not isinstance(event, ErrorEvent):
            yield None
            continue
        _clear_status_line()
        label = "Transcribe" if event.topic == "transcribe_error" else "Think"
        sys.stdout.write(f"{label} error: {event.message}\n")
        sys.stdout.flush()
        yield None


def _write_status_line(message: str) -> None:
    sys.stdout.write(f"\r\033[2K{message}")
    sys.stdout.flush()


def _clear_status_line() -> None:
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()
