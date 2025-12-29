from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


def chat_with_json_schema(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
    )
    content = response.choices[0].message.content or ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"message": content.strip()}
    if isinstance(payload, dict):
        return payload
    return {"message": content.strip()}


def chat_text(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    return content.strip()
