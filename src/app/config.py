from __future__ import annotations

from dataclasses import dataclass
from threading import Event


@dataclass(frozen=True)
class AudioChunkConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 1024
    silence_threshold: float = 0.01
    silence_duration_sec: float = 1.0
    realtime_chunk_sec: float = 2.0
    min_rms_for_transcribe: float = 0.01


@dataclass(frozen=True)
class WhisperConfig:
    main_whisper_model: str = "medium"
    realtime_whisper_model: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ja"
    beam_size: int = 1
    download_root: str = "~/data/gguf"
    denylist_phrases: tuple[str, ...] = ("ご視聴ありがとうございました",)


@dataclass(frozen=True)
class WorkerConfig:
    event_source: object
    stop_event: Event
    worker_fn: object
    publisher: object | None
    args: tuple[object, ...] = ()
