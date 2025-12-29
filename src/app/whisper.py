from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from app.config import WhisperConfig


def create_model(config: WhisperConfig, model_name: str) -> WhisperModel:
    download_root = str(Path(config.download_root).expanduser())
    return WhisperModel(
        model_name,
        device=config.device,
        compute_type=config.compute_type,
        download_root=download_root,
    )


def transcribe_audio(
    model: WhisperModel,
    audio,
    language: str,
    beam_size: int,
) -> str:
    # TODO: ご視聴ありがとうございました対策として vad_filter をあとで試す
    # https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#vad-filter
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
    )
    return "".join(segment.text for segment in segments).strip()
