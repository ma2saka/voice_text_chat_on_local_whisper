from __future__ import annotations

import time
from typing import Iterator

from app.events import ScheduleFireEvent


def cron_worker(
    _event_source,
    stop_event,
    interval_sec: float = 60.0,
    think_interval_sec: float = 30.0,
) -> Iterator[ScheduleFireEvent | None]:
    # 一定間隔で schedule_fire イベントを発行する。
    last_think_at = 0.0
    while not stop_event.is_set():
        time.sleep(interval_sec)
        if stop_event.is_set():
            break
        now = time.time()
        if now - last_think_at >= think_interval_sec:
            last_think_at = now
            yield ScheduleFireEvent(
                topic="schedule_fire",
                fired_at=now,
                schedule_type="think_update",
            )
