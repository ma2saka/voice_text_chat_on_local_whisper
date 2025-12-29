from __future__ import annotations

from app.config import WorkerConfig


def run_worker(config: WorkerConfig) -> None:
    for event in config.worker_fn(
        config.event_source, config.stop_event, *config.args
    ):
        if config.stop_event.is_set():
            break
        if event is None or config.publisher is None:
            continue
        config.publisher.publish(event)
