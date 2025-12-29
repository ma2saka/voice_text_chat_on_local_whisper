from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event
from typing import Iterator, Literal

import numpy as np
import sounddevice as sd

from app.config import AudioChunkConfig
from app.events import AudioChunkEvent


ChunkKind = Literal["split", "realtime"]


@dataclass(frozen=True)
class AudioChunk:
    kind: ChunkKind
    index: int
    samples: np.ndarray
    sample_rate: int


@dataclass
class ChunkState:
    current_buffers: list[np.ndarray] = field(default_factory=list)
    silence_buffers: list[np.ndarray] = field(default_factory=list)
    realtime_buffers: list[np.ndarray] = field(default_factory=list)
    silence_frames: int = 0
    realtime_frames: int = 0
    split_count: int = 0
    realtime_count: int = 0


class ChunkProcessor:
    def __init__(self, config: AudioChunkConfig) -> None:
        self._config = config
        self._state = ChunkState()
        self._realtime_chunk_frames = int(
            config.realtime_chunk_sec * config.sample_rate
        )
        self._silence_frames_to_split = int(
            config.silence_duration_sec * config.sample_rate
        )

    def process_block(self, samples: np.ndarray, frames: int) -> list[AudioChunk]:
        rms = float(np.sqrt(np.mean(np.square(samples))))
        state = self._state
        sample_copy = samples.copy()
        state.current_buffers.append(sample_copy)
        state.realtime_buffers.append(sample_copy)
        state.realtime_frames += frames
        if rms < self._config.silence_threshold:
            state.silence_frames += frames
            state.silence_buffers.append(sample_copy)
        else:
            state.silence_frames = 0
            state.silence_buffers.clear()
        events: list[AudioChunk] = []
        if state.realtime_frames >= self._realtime_chunk_frames:
            buffer = np.concatenate(state.realtime_buffers)
            while buffer.shape[0] >= self._realtime_chunk_frames:
                chunk_samples = buffer[: self._realtime_chunk_frames]
                buffer = buffer[self._realtime_chunk_frames :]
                state.realtime_count += 1
                events.append(
                    AudioChunk(
                        kind="realtime",
                        index=state.realtime_count,
                        samples=chunk_samples,
                        sample_rate=self._config.sample_rate,
                    )
                )
            state.realtime_buffers = [buffer] if buffer.size else []
            state.realtime_frames = buffer.shape[0]
        if state.silence_frames >= self._silence_frames_to_split:
            if state.silence_buffers:
                buffers = state.current_buffers[
                    : -len(state.silence_buffers)
                ]
            else:
                buffers = state.current_buffers
            if buffers:
                chunk_samples = np.concatenate(buffers)
                state.split_count += 1
                events.append(
                    AudioChunk(
                        kind="split",
                        index=state.split_count,
                        samples=chunk_samples,
                        sample_rate=self._config.sample_rate,
                    )
                )
            state.current_buffers.clear()
            state.silence_buffers.clear()
            state.silence_frames = 0
        return events


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray
    frames: int


class MicrophoneEventSource:
    def __init__(self, config: AudioChunkConfig, stop_event: Event) -> None:
        self._config = config
        self._stop_event = stop_event
        self._queue: Queue[AudioBlock | str] = Queue()

    def stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            blocksize=self._config.block_size,
            dtype="float32",
            callback=self._on_audio,
        )

    def pick(self, topic: str, timeout: float) -> AudioBlock | None:
        if topic != "audio_block":
            return None
        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            return None
        if isinstance(item, str):
            return None
        return item

    def _on_audio(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if self._stop_event.is_set():
            return
        if status:
            self._queue.put(str(status))
        if indata.size == 0:
            return
        samples = indata
        if samples.ndim > 1:
            samples = samples[:, 0]
        self._queue.put(AudioBlock(samples=samples.copy(), frames=frames))


def listener_worker(
    event_source: MicrophoneEventSource,
    stop_event: Event,
    processor: ChunkProcessor,
) -> Iterator[AudioChunkEvent | None]:
    # マイク入力を受け取り、チャンク化イベントを発行する。
    with event_source.stream():
        while not stop_event.is_set():
            block = event_source.pick("audio_block", timeout=0.2)
            if not block:
                yield None
                continue
            for chunk in processor.process_block(block.samples, block.frames):
                yield AudioChunkEvent(topic="audio_chunk", chunk=chunk)
