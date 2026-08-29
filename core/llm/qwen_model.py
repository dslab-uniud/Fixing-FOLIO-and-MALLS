import json
import os
import re
import warnings
from typing import Type

from pydantic import BaseModel

from .base import LLM, Message

# Official Qwen API base URL (OpenAI-compatible endpoint via DashScope)
_QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenModel(LLM):
    """LLM backend for Qwen models, supporting two execution modes.

    mode="local"
        Loads the model locally via HuggingFace ``transformers``
        (AutoModelForCausalLM + AutoTokenizer).  Logprobs are extracted from
        ``output.scores`` / ``transition_scores``.  Structured output is
        achieved by appending a JSON-schema instruction to the prompt and
        validating the response with Pydantic.

    mode="api"
        Calls a Qwen-hosted or self-hosted (e.g. vLLM) endpoint that exposes
        an OpenAI-compatible Chat Completions API.

    API key resolution order (api mode only):
        1. `api_key` constructor argument
        2. content of `api_key_path` file (reads QWEN_API_KEY=... line)
        3. ``QWEN_API_KEY`` environment variable
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "local",
        api_key: str | None = None,
        api_key_path: str | None = None,
        api_base_url: str | None = None,
        device: str = "cuda",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
    ):
        """
        Args:
            model_name: HuggingFace model ID (local) or API model name,
                e.g. "Qwen/Qwen2.5-7B-Instruct" or "qwen-max".
            mode: "local" (transformers) or "api" (OpenAI-compatible endpoint).
            api_key: API key string for api mode.
            api_key_path: path to file containing ``QWEN_API_KEY=<value>``.
            api_base_url: base URL for the API endpoint; defaults to the
                official Qwen API URL when None in api mode.
            device: torch device string for local mode, e.g. "cuda", "cpu",
                "cuda:1".
            temperature: sampling temperature.
            max_new_tokens: token budget for the generated completion.
        """
        if mode not in ("local", "api"):
            raise ValueError(f"mode must be 'local' or 'api', got {mode!r}")

        self.model_name = model_name
        self.mode = mode
        self.api_base_url = api_base_url or _QWEN_API_BASE_URL
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Lazy-loaded in local mode
        self._model = None
        self._tokenizer = None

        self._api_key = self._load_api_key(api_key, api_key_path)

        if mode == "api":
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key, base_url=self.api_base_url
            )
        else:
            self._client = None

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
        """Prompt Qwen and return the response.

        Dispatches to ``_generate_local`` or ``_generate_api`` based on
        ``self.mode``.  The ``online`` parameter is accepted for interface
        compatibility but batch mode is not natively supported for Qwen; a
        warning is emitted if online=False is passed in local mode.

        Args:
            messages: conversation history or single user prompt string.
            online: True → real-time; False → batch (api mode only).
            response_format: Pydantic class for structured output.
            logprobs: request token-level log-probabilities.
            top_logprobs: number of top-token alternatives per position.
            **kwargs: per-call overrides (temperature, max_new_tokens, etc.).

        Returns:
            Same return semantics as OpenAIModel.generate().
        """
        if self.mode == "local":
            if not online:
                warnings.warn(
                    "Batch mode (online=False) is not supported for local Qwen; "
                    "running in online mode instead.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._generate_local(
                messages, response_format, logprobs, top_logprobs, **kwargs
            )
        else:
            return self._generate_api(
                messages, response_format, logprobs, top_logprobs, **kwargs
            )

    # ------------------------------------------------------------------
    # Private: local (HuggingFace transformers)
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        """Lazy-load the HuggingFace model and tokenizer.

        Called on first use so that importing the module does not trigger
        a heavy model download.  Sets ``self._model`` and ``self._tokenizer``.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _generate_local(
        self,
        messages: list[Message] | str,
        response_format: Type[BaseModel] | None,
        logprobs: bool,
        top_logprobs: int | None,
        **kwargs,
    ) -> str | dict:
        """Run generation locally via transformers.

        When response_format is set, a JSON-schema instruction is appended to
        the last user message; the raw output is parsed with
        ``_parse_structured_local``.

        When logprobs=True, ``generate()`` is called with
        ``output_scores=True, return_dict_in_generate=True`` and token
        probabilities are extracted via ``_extract_logprobs_local``.
        """
        import torch

        if self._model is None:
            self._load_local_model()

        temperature = kwargs.pop("temperature", self.temperature)
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

        # Build chat messages list
        if isinstance(messages, str):
            chat_messages = [{"role": "user", "content": messages}]
        else:
            chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Append JSON-schema instruction for structured output
        if response_format is not None:
            schema_str = json.dumps(response_format.model_json_schema(), indent=2)
            json_instruction = (
                f"\n\nYou MUST respond with a valid JSON object that matches "
                f"this schema:\n{schema_str}\nReturn only the JSON object, no other text."
            )
            if chat_messages and chat_messages[-1]["role"] == "user":
                chat_messages[-1] = dict(chat_messages[-1])
                chat_messages[-1]["content"] += json_instruction
            else:
                chat_messages.append({"role": "user", "content": json_instruction})

        # Apply Qwen chat template
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        if logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        if logprobs:
            generated_ids = outputs.sequences[:, input_length:]
            output_text = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            logprob_list = self._extract_logprobs_local(
                outputs, input_length, top_logprobs
            )
        else:
            generated_ids = outputs[:, input_length:]
            output_text = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            logprob_list = None

        result: dict = {}

        if response_format is not None:
            result["structured"] = self._parse_structured_local(
                output_text, response_format
            )

        if logprobs:
            result["text"] = output_text
            result["logprobs"] = logprob_list

        # Simplify: if only plain text is needed return a bare string
        if not result:
            return output_text

        # If structured-only (no logprobs) keep "text" out per the base contract
        if response_format is not None and not logprobs:
            return result  # {"structured": ...}

        return result

    def _parse_structured_local(
        self, output_text: str, schema_class: Type[BaseModel]
    ) -> dict:
        """Extract and validate JSON from the model's free-form output.

        Attempts to locate a JSON block in ``output_text``, parse it, and
        validate it against ``schema_class``.  Returns the validated dict
        (model_dump()).

        Raises:
            ValueError: if no valid JSON matching the schema is found.
        """
        # Try a fenced ```json ... ``` block first
        fenced = re.search(r"```json\s*(.*?)\s*```", output_text, re.DOTALL)
        if fenced:
            json_str = fenced.group(1)
        else:
            # Fall back to the first {...} block in the output
            raw = re.search(r"\{.*\}", output_text, re.DOTALL)
            if not raw:
                raise ValueError(
                    f"No JSON object found in model output:\n{output_text!r}"
                )
            json_str = raw.group(0)

        data = json.loads(json_str)
        validated = schema_class.model_validate(data)
        return validated.model_dump()

    def _extract_logprobs_local(
        self,
        outputs,
        input_length: int,
        top_k: int | None,
    ) -> list[dict]:
        """Convert HuggingFace generation scores to log-probability dicts.

        Args:
            outputs: the ``GenerateOutput`` object returned by
                ``model.generate(output_scores=True, return_dict_in_generate=True)``.
            input_length: number of input tokens (to offset into sequences).
            top_k: number of top-token alternatives to include per position;
                None means only the chosen token is reported.

        Returns:
            List of dicts per generated token, each with:
                - "token": decoded token string
                - "logprob": log-probability of the chosen token
                - "top_logprobs": top-k alternatives (empty list if top_k is None)
        """
        import torch

        result = []
        for step_idx, scores in enumerate(outputs.scores):
            # scores[step_idx]: tensor of shape (batch_size, vocab_size)
            log_probs = torch.nn.functional.log_softmax(scores[0], dim=-1)
            chosen_token_id = outputs.sequences[0, input_length + step_idx].item()
            chosen_logprob = log_probs[chosen_token_id].item()
            token_str = self._tokenizer.decode([chosen_token_id])

            entry = {
                "token": token_str,
                "logprob": chosen_logprob,
                "top_logprobs": [],
            }

            if top_k is not None and top_k > 0:
                topk_logprobs, topk_ids = torch.topk(log_probs, k=top_k)
                entry["top_logprobs"] = [
                    {
                        "token": self._tokenizer.decode([tid.item()]),
                        "logprob": lp.item(),
                    }
                    for tid, lp in zip(topk_ids, topk_logprobs)
                ]

            result.append(entry)

        return result

    # ------------------------------------------------------------------
    # Private: API (OpenAI-compatible endpoint)
    # ------------------------------------------------------------------

    def _generate_api(
        self,
        messages: list[Message] | str,
        response_format: Type[BaseModel] | None,
        logprobs: bool,
        top_logprobs: int | None,
        **kwargs,
    ) -> str | dict:
        """Call the Qwen API via an OpenAI-compatible client.

        Structured output is requested using the same
        ``type_to_response_format_param`` approach as OpenAIModel when
        supported by the endpoint; otherwise falls back to prompt-based JSON.
        """
        try:
            from openai.lib.parsing._completions import type_to_response_format_param
        except ImportError:
            # openai >= 2.0 removed this private module; use pydantic_function_tool instead
            def type_to_response_format_param(schema_class):
                import json
                return {"type": "json_schema", "json_schema": {"name": schema_class.__name__, "schema": schema_class.model_json_schema()}}

        if isinstance(messages, str):
            openai_messages = [{"role": "user", "content": messages}]
        else:
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

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
            try:
                call_kwargs["response_format"] = type_to_response_format_param(
                    response_format
                )
            except Exception:
                # Endpoint may not support JSON-schema response_format;
                # fall back to prompt-based instruction.
                schema_str = json.dumps(
                    response_format.model_json_schema(), indent=2
                )
                json_instruction = (
                    f"\n\nYou MUST respond with a valid JSON object matching "
                    f"this schema:\n{schema_str}\nReturn only the JSON, no other text."
                )
                openai_messages = list(openai_messages)
                if openai_messages and openai_messages[-1]["role"] == "user":
                    openai_messages[-1] = dict(openai_messages[-1])
                    openai_messages[-1]["content"] += json_instruction
                else:
                    openai_messages.append(
                        {"role": "user", "content": json_instruction}
                    )
                call_kwargs["messages"] = openai_messages

        response = self._client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        result: dict = {}

        if response_format is not None:
            result["structured"] = self._parse_structured_local(
                content, response_format
            )

        if logprobs:
            result["text"] = content
            result["logprobs"] = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top_logprobs": [
                        {"token": x.token, "logprob": x.logprob}
                        for x in (t.top_logprobs or [])
                    ],
                }
                for t in response.choices[0].logprobs.content
            ]

        if not result:
            return content

        if response_format is not None and not logprobs:
            return result  # {"structured": ...}

        return result

    def _load_api_key(
        self, api_key: str | None, api_key_path: str | None
    ) -> str | None:
        """Resolve the Qwen API key (api mode only).

        Returns None if mode="local" and no key was provided.
        """
        if api_key:
            return api_key

        if api_key_path:
            with open(api_key_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("QWEN_API_KEY="):
                        return line.split("=", 1)[1].strip()

        key = os.environ.get("QWEN_API_KEY")
        if key:
            return key

        if self.mode == "api":
            raise ValueError(
                "No Qwen API key found. Provide api_key, api_key_path, "
                "or set the QWEN_API_KEY environment variable."
            )
        return None
