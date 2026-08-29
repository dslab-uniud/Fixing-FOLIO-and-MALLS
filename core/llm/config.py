"""YAML-based factory for LLM instances.

Expected YAML structure (see configs/models.yaml for a full example)::

    default:
      type: openai
      model_name: gpt-4o
      api_key_path: secrets/api_keys.env
      temperature: 0.0
      max_tokens: 1024

    qwen_local:
      type: qwen
      model_name: Qwen/Qwen2.5-7B-Instruct
      mode: local
      device: cuda
      temperature: 0.0
      max_new_tokens: 1024

Supported ``type`` values: ``"openai"``, ``"qwen"``.
"""

import yaml

from .base import LLM
from .openai_model import OpenAIModel
from .qwen_model import QwenModel

_BACKENDS = {
    "openai": OpenAIModel,
    "qwen": QwenModel,
}


def load_model_from_config(
    config_path: str, model_key: str = "default"
) -> LLM:
    """Instantiate an LLM from a YAML configuration file.

    Args:
        config_path: path to the YAML file (e.g. "configs/models.yaml").
        model_key: the top-level key in the YAML to load.  Defaults to
            "default".

    Returns:
        A fully initialised LLM subclass instance (OpenAIModel or QwenModel).

    Raises:
        KeyError: if ``model_key`` is not found in the config file.
        ValueError: if ``type`` is not a recognised backend name.
        FileNotFoundError: if ``config_path`` does not exist.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if model_key not in config:
        available = sorted(config.keys())
        raise KeyError(
            f"Model key {model_key!r} not found in {config_path!r}. "
            f"Available keys: {available}"
        )

    params = dict(config[model_key])
    model_type = params.pop("type")

    if model_type not in _BACKENDS:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Supported types: {sorted(_BACKENDS.keys())}"
        )

    return _BACKENDS[model_type](**params)


def list_available_models(config_path: str) -> list[str]:
    """Return the list of model keys defined in a YAML config file.

    Args:
        config_path: path to the YAML file.

    Returns:
        Sorted list of top-level key names, e.g. ["default", "qwen_local"].
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return sorted(config.keys())
