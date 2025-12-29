# Maintenance Instructions

> Note: GitHub Copilot reads instructions from `.github/copilot-instructions.md` (if present). This file is for humans/maintainers. If you want Copilot to use these rules, copy or move the relevant sections into `.github/copilot-instructions.md`.

## Project Overview
- Microphone audio capture → chunking → transcription (faster-whisper) → chat response (OpenAI) → display.
- Event-driven, broadcast-style queues with per-subscriber retention.
- Logging goes to `logs/app-YYYYMMDD-HH.log` as JSON.

## Run
- Start: `uv run -m app`
- Requires `.env` with `OPENAI_API_KEY`.

## Key Architecture
- **Event broker**: `src/app/events.py`
- **Worker runner**: `src/app/worker.py`
- **Bootstrap**: `src/app/bootstrap.py` (dotenv + CUDA runtime prep)
- **Core entry**: `src/app/app.py`
- **Workers**: `src/app/workers/*`
  - `listener_worker`: mic input → `AudioChunkEvent`
  - `transcribe_worker`: chunk → `TranscriptionEvent`
  - `chat_worker`: split transcription → `AssistantEvent`
  - `display` workers: terminal output only
  - `cron_worker`: schedule_fire events
  - `think_worker`: updates `context.thinking`
  - `monitor_worker`: logs queue sizes (JSON)

## Configuration
- `src/app/config.py`
  - `AudioChunkConfig`: input rates, silence detection, realtime chunk seconds
  - `WhisperConfig`: model/device/compute_type + `denylist_phrases`
  - `WorkerConfig`: thread runner config

## GPU Setup
- Uses CUDA libs from `nvidia-cudnn-cu12` / `nvidia-cublas-cu12`.
- `bootstrap.prepare_cuda_runtime()` preloads libs to avoid runtime failures.

## Logging
- JSON logs only
- Low-RMS skip logs are DEBUG level.

## Display
- User: bright cyan
- Assistant: green
- System: yellow
- User metadata `(transcribe/chunk)` is gray

## Thinking
- `context.thinking` is included as `role=system` when calling OpenAI.
- `think_worker` runs on `schedule_fire` (`think_update`) and logs JSON:
  - `{ "event": "thinking_update", "thinking": "..." }`

## JSON Response (Assistant)
- OpenAI response uses JSON schema:
  - `{ "message": "..." }`
- Display shows only `message` content.

## Common Maintenance Tasks
- Add a new worker:
  1) Implement generator-style worker function.
  2) Subscribe to an event topic in `app.py`.
  3) Register the worker thread with a `WorkerConfig`.
- Change schedule:
  - `cron_worker(interval_sec, think_interval_sec)` in `app.py`.

## Notes
- If you need Copilot-specific instructions, create `.github/copilot-instructions.md`.

## Architecture Principles (Must Follow)
- **Event-driven, broadcast-first**: Producers publish events; consumers subscribe with their own queues. Do not make a worker call another worker directly.
- **Single responsibility**: Each worker should do one job (capture, transcribe, chat, display, monitor, schedule).
- **Queue isolation**: Use per-subscriber queues and keep them unbounded; do not drop items implicitly.
- **Generator workers**: Worker functions should be generator-style (`while not stop_event`, `yield event or None`) to standardize the runner loop.
- **No UI in core workers**: Only display workers write to stdout; keep processing workers pure.
- **State updates are explicit**: Context updates (messages/thinking) happen in designated workers (chat/think), not in display or monitor.
