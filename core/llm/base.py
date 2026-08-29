from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .chat_session import ChatSession


@dataclass
class Message:
    """A single message in a conversation.

    role: "system" | "user" | "assistant"
    content: the text content of the message
    """
    role: str
    content: str


class LLM(ABC):
    """Abstract base class for all LLM backends.

    Subclasses must implement `generate()`. The `chat()` convenience method
    returns a ChatSession bound to this instance for multi-turn interactions.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[Message] | str,
        online: bool = True,
        response_format: Type[BaseModel] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> str | dict:
        """Prompt the LLM and return its response.

        Args:
            messages:
                A single string (auto-wrapped as a user message) or a list of
                Message objects representing the full conversation so far.
            online:
                If True, call the model in real-time and return immediately.
                If False, submit the request for batch processing (where
                supported by the backend).
            response_format:
                A Pydantic BaseModel subclass that defines the expected output
                structure.  When provided the method returns a dict
                (model_dump() of the parsed response); otherwise returns a str.
                For OpenAI this uses type_to_response_format_param(); for local
                Qwen it uses prompt-based JSON extraction + Pydantic validation.
            logprobs:
                If True, also return token-level log-probabilities.  The return
                value becomes a dict with keys "text" (str | dict) and
                "logprobs" (list[dict]).
            top_logprobs:
                Number of top-token alternatives to return per position.
                Ignored when logprobs=False.
            **kwargs:
                Additional backend-specific parameters (e.g. temperature,
                max_tokens overrides for a single call).

        Returns:
            - str: plain text response (default)
            - dict with key "structured": parsed Pydantic model (response_format set)
            - dict with keys "text" and "logprobs": (logprobs=True)
            - dict with keys "text", "logprobs", "structured": (both set)
        """
        pass

    def chat(self, system_prompt: str | None = None) -> "ChatSession":
        """Return a new ChatSession bound to this LLM instance.

        Args:
            system_prompt: optional system message prepended to every session.
        """
        from .chat_session import ChatSession
        return ChatSession(llm=self, system_prompt=system_prompt)
