import json
from typing import Type, TYPE_CHECKING

from pydantic import BaseModel

from .base import Message

if TYPE_CHECKING:
    from .base import LLM


class ChatSession:
    """Manages a stateful multi-turn conversation with an LLM.

    The session maintains a growing history of Message objects.  Each call to
    `send()` appends the user message, calls the underlying LLM, appends the
    assistant reply, and returns the response — so context is automatically
    preserved across turns.

    Usage example::

        session = llm.chat(system_prompt="You are a FOL expert.")
        answer1 = session.send("Is ∀x P(x) equivalent to ∃x P(x)?")
        answer2 = session.send("Why not?")   # model still has context
    """

    def __init__(self, llm: "LLM", system_prompt: str | None = None):
        """Initialise a session.

        Args:
            llm: the LLM backend to use for generation.
            system_prompt: optional system message; if provided it is prepended
                to the history and preserved across resets (unless
                keep_system=False is passed to reset()).
        """
        self._llm = llm
        self._system_prompt = system_prompt
        self._history: list[Message] = []
        if system_prompt is not None:
            self._history.append(Message(role="system", content=system_prompt))

    def send(
        self,
        user_message: str,
        response_format: Type[BaseModel] | None = None,
        logprobs: bool = False,
        **kwargs,
    ) -> str | dict:
        """Send a user message and return the model's reply.

        The user message and the assistant reply are both appended to the
        internal history, so subsequent calls see the full conversation.

        Args:
            user_message: the text to send as the next user turn.
            response_format: Pydantic class for structured output (passed
                through to LLM.generate()).
            logprobs: whether to request token log-probabilities.
            **kwargs: extra arguments forwarded to LLM.generate().

        Returns:
            The assistant's response (str or dict, same semantics as
            LLM.generate()).
        """
        self._history.append(Message(role="user", content=user_message))

        response = self._llm.generate(
            messages=list(self._history),
            response_format=response_format,
            logprobs=logprobs,
            **kwargs,
        )

        # Extract a plain-text representation to store in history so the model
        # sees its own previous replies in subsequent turns.
        if isinstance(response, dict):
            assistant_text = response.get("text", "")
            if not assistant_text and "structured" in response:
                assistant_text = json.dumps(response["structured"])
        else:
            assistant_text = response

        self._history.append(Message(role="assistant", content=assistant_text))
        return response

    def reset(self, keep_system: bool = True) -> None:
        """Clear the conversation history.

        Args:
            keep_system: if True (default), the initial system message is
                retained; if False, the history is fully erased.
        """
        if keep_system and self._system_prompt is not None:
            self._history = [Message(role="system", content=self._system_prompt)]
        else:
            self._history = []

    def get_history(self) -> list[Message]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    def inject(self, role: str, content: str) -> None:
        """Manually insert a message into the history.

        Useful for injecting few-shot examples or tool outputs before the
        next `send()` call.

        Args:
            role: "system", "user", or "assistant".
            content: the message text.
        """
        self._history.append(Message(role=role, content=content))
