from __future__ import annotations

import json
import time
from typing import Iterator

from app.events import EventBroker


def monitor_worker(
    broker: EventBroker,
    stop_event,
    logger,
    interval_sec: float = 5.0,
) -> Iterator[None]:
    # 各キューのサイズを定期的にログ出力する。
    while not stop_event.is_set():
        sizes = broker.queue_sizes()
        if sizes:
            logger.info(
                json.dumps(
                    {
                        "event": "queue_sizes",
                        "queues": [
                            {"topic": topic, "size": size} for topic, size in sizes
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        time.sleep(interval_sec)
        yield None
