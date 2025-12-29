from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class Context:
    system_prompt: str
    thinking: str | None = None
    messages: list[Message] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text))

    def add_system_message(self, text: str) -> None:
        self.messages.append(Message(role="system", content=text))

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(Message(role="assistant", content=text))

    def set_thinking(self, text: str | None) -> None:
        self.thinking = text

    def to_dict(self) -> dict[str, object]:
        return {
            "system_prompt": self.system_prompt,
            "thinking": self.thinking,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ],
        }

    def to_openai_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.thinking:
            messages.append({"role": "system", "content": self.thinking})
        for message in self.messages:
            messages.append(
                {"role": message.role, "content": message.content}
            )
        return messages
