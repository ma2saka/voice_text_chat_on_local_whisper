from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Iterator
from queue import Empty, Queue

import ctranslate2
import numpy as np

from app.config import WhisperConfig
from app.events import AudioChunkEvent, ErrorEvent, TranscriptionEvent
from app.workers.listner import AudioChunk
from app.whisper import create_model, transcribe_audio


@dataclass(frozen=True)
class Transcription:
    kind: str
    index: int
    text: str


class TranscribeWorker:
    def __init__(self, config: WhisperConfig) -> None:
        self._config = config
        if config.device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            raise RuntimeError("CUDA device not found for faster-whisper.")
        self._models: dict[str, object] = {}
        for model_name in {
            config.main_whisper_model,
            config.realtime_whisper_model,
        }:
            self._models[model_name] = create_model(
                config,
                model_name,
            )

    def transcribe(self, chunk: AudioChunk) -> Transcription:
        audio = self._normalize_audio(chunk.samples)
        model_name = (
            self._config.main_whisper_model
            if chunk.kind == "split"
            else self._config.realtime_whisper_model
        )
        text = transcribe_audio(
            self._models[model_name],
            audio,
            language=self._config.language,
            beam_size=self._config.beam_size,
        )
        return Transcription(kind=chunk.kind, index=chunk.index, text=text)

    @staticmethod
    def _normalize_audio(samples: np.ndarray) -> np.ndarray:
        if samples.dtype != np.float32:
            return samples.astype(np.float32)
        return samples


def transcribe_worker(
    input_queue: Queue[object],
    stop_event,
    transcriber: TranscribeWorker,
    audio_config,
    whisper_config: WhisperConfig,
    logger,
) -> Iterator[TranscriptionEvent | ErrorEvent | None]:
    # 音声チャンクを受け取り、書き起こし結果イベントのみを発行する。
    while not stop_event.is_set():
        try:
            event = input_queue.get(timeout=0.2)
        except Empty:
            yield None
            continue
        chunk_event = event
        if not isinstance(chunk_event, AudioChunkEvent):
            yield None
            continue
        chunk: AudioChunk = chunk_event.chunk
        rms = float(np.sqrt(np.mean(np.square(chunk.samples))))
        if rms < audio_config.min_rms_for_transcribe:
            payload = {
                "event": {
                    "kind": chunk.kind,
                    "index": chunk.index,
                    "sample_rate": chunk.sample_rate,
                    "frames": int(chunk.samples.shape[0]),
                    "duration_sec": chunk.samples.shape[0]
                    / float(chunk.sample_rate),
                },
                "transcribe": {
                    "text": "",
                    "skipped": True,
                    "reason": "low_rms",
                },
            }
            logger.debug(json.dumps(payload, ensure_ascii=False))
            yield None
            continue
        start_time = time.perf_counter()
        try:
            result = transcriber.transcribe(chunk)
        except Exception as exc:
            logger.info(
                json.dumps(
                    {
                        "event": {
                            "kind": chunk.kind,
                            "index": chunk.index,
                        },
                        "transcribe": {
                            "error": str(exc),
                        },
                    },
                    ensure_ascii=False,
                )
            )
            yield ErrorEvent(
                topic="transcribe_error",
                message=str(exc),
            )
            continue
        transcribe_sec = time.perf_counter() - start_time
        payload = {
            "event": {
                "kind": chunk.kind,
                "index": chunk.index,
                "sample_rate": chunk.sample_rate,
                "frames": int(chunk.samples.shape[0]),
                "duration_sec": chunk.samples.shape[0]
                / float(chunk.sample_rate),
            },
            "transcribe": {
                "text": result.text,
            },
        }
        logger.info(json.dumps(payload, ensure_ascii=False))
        if _is_denied_text(result.text, whisper_config.denylist_phrases):
            yield None
            continue
        chunk_sec = chunk.samples.shape[0] / float(chunk.sample_rate)
        if chunk.kind == "realtime":
            yield TranscriptionEvent(
                topic="realtime_transcription",
                kind=chunk.kind,
                text=result.text,
                transcribe_sec=transcribe_sec,
                chunk_sec=chunk_sec,
                chunk_index=chunk.index,
            )
            continue
        if chunk.kind == "split":
            yield TranscriptionEvent(
                topic="split_transcription",
                kind=chunk.kind,
                text=result.text,
                transcribe_sec=transcribe_sec,
                chunk_sec=chunk_sec,
                chunk_index=chunk.index,
            )


def _is_denied_text(text: str, denylist: tuple[str, ...]) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    return normalized in denylist
