import importlib
from typing import Dict
from .base import AsyncTranslator
from .registry import list_providers, resolve_provider_name, get_provider_factory


def _load_custom_provider(
    path: str, options: Dict[str, str]
) -> AsyncTranslator:
    if ":" not in path:
        raise ValueError(
            "Custom providers must use module path syntax like "
            "`package.module:ClassName`."
        )
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name, None)
    if provider_cls is None:
        raise ValueError(
            f"Provider class `{class_name}` not found in `{module_path}`."
        )
    return provider_cls(**options)


def create_translator(
    provider: str, options: Dict[str, str]
) -> AsyncTranslator:
    resolved = resolve_provider_name(provider)
    factory = get_provider_factory(resolved)
    if factory:
        return factory(options)
    if ":" in provider:
        return _load_custom_provider(provider, options)
    available = ", ".join(list_providers())
    raise ValueError(
        f"Unknown provider `{provider}`. Available providers: {available}."
    )
