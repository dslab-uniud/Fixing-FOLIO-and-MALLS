import json
import os
import tempfile
import time
from typing import Type

from pydantic import BaseModel

from .base import LLM, Message


class OpenAIModel(LLM):
    """LLM backend for OpenAI-hosted models (GPT-4o, GPT-4o-mini, etc.).

    API key resolution order (first match wins):
        1. `api_key` constructor argument
        2. content of `api_key_path` file (reads OPENAI_API_KEY=... line)
        3. ``OPENAI_API_KEY`` environment variable

    Structured output uses ``openai.lib.parsing._completions.type_to_response_format_param``
    so the schema is derived directly from a Pydantic BaseModel subclass.

    Logprobs are requested via the standard ``logprobs`` / ``top_logprobs``
    parameters of the Chat Completions API.

    Batch mode (``online=False``) submits requests through the OpenAI Batch API
    and returns a ``batch_id`` string; use ``retrieve_batch(batch_id)`` to
    collect results once the job is complete.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_key_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Args:
            model_name: OpenAI model identifier, e.g. "gpt-4o" or "gpt-4o-mini".
            api_key: API key string (not recommended for scripts — prefer a file).
            api_key_path: path to a file containing ``OPENAI_API_KEY=<value>``.
            temperature: sampling temperature (0.0 = greedy / deterministic).
            max_tokens: maximum number of tokens in the completion.
        """
        import openai

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self._load_api_key(api_key, api_key_path)
        self._client = openai.OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[Message] | str,
        online: bool = True,
        response_format: Type[BaseModel] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> str | dict:
        """Call the OpenAI Chat Completions (or Batch) API.

        Args:
            messages: conversation history or a single user prompt string.
            online: True → real-time completion; False → submit to Batch API
                (returns a dict with key "batch_id").
            response_format: Pydantic class; when set, structured output is
                requested via ``type_to_response_format_param`` and the return
                value includes a "structured" key with the parsed dict.
            logprobs: request token-level log-probabilities from the API.
            top_logprobs: how many top-token alternatives per position (1–20).
            **kwargs: per-call overrides (e.g. temperature, max_tokens).

        Returns:
            - str: plain text (default)
            - dict{"structured": ...}: structured output only
            - dict{"text": ..., "logprobs": [...]}: logprobs only
            - dict{"text": ..., "logprobs": [...], "structured": ...}: both
            - dict{"batch_id": ...}: when online=False
        """
        try:
            from openai.lib.parsing._completions import type_to_response_format_param
        except ImportError:
            from openai.lib._parsing._completions import type_to_response_format_param

        openai_messages = self._messages_to_openai_format(messages)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        if not online:
            request = {
                "custom_id": "req-0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": openai_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }
            batch_id = self.submit_batch([request])
            return {"batch_id": batch_id}

        call_kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if logprobs:
            call_kwargs["logprobs"] = True
            if top_logprobs is not None:
                call_kwargs["top_logprobs"] = top_logprobs

        if response_format is not None:
            call_kwargs["response_format"] = type_to_response_format_param(response_format)

        response = self._client.chat.completions.create(**call_kwargs)

        content = response.choices[0].message.content
        result = {}

        if response_format is not None:
            parsed = response_format.model_validate_json(content)
            result["structured"] = parsed.model_dump()

        if logprobs:
            result["text"] = content
            result["logprobs"] = self._parse_logprobs(response)
            if response_format is not None:
                # already have "structured"; add "text" if not already present
                pass
        elif response_format is None:
            # plain text only — return a bare string
            return content

        # If only structured (no logprobs), return just the structured dict
        if response_format is not None and not logprobs:
            return result  # {"structured": ...}

        return result

    # ------------------------------------------------------------------
    # Batch API helpers
    # ------------------------------------------------------------------

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a list of chat-completion requests as an OpenAI Batch job.

        Args:
            requests: list of request dicts, each following the OpenAI batch
                input format (``custom_id``, ``method``, ``url``, ``body``).

        Returns:
            The ``batch_id`` string assigned by OpenAI.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                file_obj = self._client.files.create(file=f, purpose="batch")
            batch = self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            return batch.id
        finally:
            os.unlink(tmp_path)

    def retrieve_batch(self, batch_id: str) -> list[dict]:
        """Poll and retrieve results from a completed Batch API job.

        Args:
            batch_id: the ID returned by ``submit_batch``.

        Returns:
            List of result dicts, one per original request, each containing
            the completion or an error description.
        """
        while True:
            batch = self._client.batches.retrieve(batch_id)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(30)

        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} ended with status: {batch.status}"
            )

        output_file = self._client.files.content(batch.output_file_id)
        results = []
        for line in output_file.text.splitlines():
            if line.strip():
                results.append(json.loads(line))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_api_key(
        self, api_key: str | None, api_key_path: str | None
    ) -> str:
        """Resolve the API key from argument > explicit file > env var > .env file.

        Search order:
        1. ``api_key`` constructor argument
        2. file at ``api_key_path`` (reads ``OPENAI_API_KEY=...`` line)
        3. ``OPENAI_API_KEY`` environment variable
        4. ``.env`` file at the project root (``OPENAI_API_KEY=...`` line)

        Raises:
            ValueError: if no key is found from any source.
        """
        if api_key:
            return api_key

        if api_key_path:
            with open(api_key_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip()

        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key

        # Auto-detect .env in the project root (walk up from this file's location)
        env_path = self._find_dotenv()
        if env_path:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        val = line.split("=", 1)[1].strip()
                        if val and not val.startswith("sk-your"):
                            return val

        raise ValueError(
            "No OpenAI API key found. Options:\n"
            "  1. Set OPENAI_API_KEY in the .env file at the project root\n"
            "  2. Set the OPENAI_API_KEY environment variable\n"
            "  3. Pass api_key= or api_key_path= to OpenAIModel()"
        )

    @staticmethod
    def _find_dotenv() -> str | None:
        """Walk up from this file's directory to find the nearest .env file."""
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):  # max 6 levels up
            candidate = os.path.join(current, ".env")
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    def _messages_to_openai_format(
        self, messages: list[Message] | str
    ) -> list[dict]:
        """Convert Message objects (or a plain string) to OpenAI's message list.

        A plain string is wrapped as ``{"role": "user", "content": <str>}``.
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return [{"role": m.role, "content": m.content} for m in messages]

    def _parse_logprobs(self, response) -> list[dict]:
        """Extract token log-probability data from a Chat Completions response.

        Returns:
            List of dicts, one per output token, each with keys:
                - "token": the token string
                - "logprob": the log-probability (float)
                - "top_logprobs": list of {token, logprob} for alternatives
        """
        token_logprobs = response.choices[0].logprobs.content
        result = []
        for token_lp in token_logprobs:
            entry = {
                "token": token_lp.token,
                "logprob": token_lp.logprob,
                "top_logprobs": [
                    {"token": t.token, "logprob": t.logprob}
                    for t in (token_lp.top_logprobs or [])
                ],
            }
            result.append(entry)
        return result
