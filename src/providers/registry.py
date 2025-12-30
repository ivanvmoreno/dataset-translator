from typing import Callable, Dict, List, Optional
from .base import ProviderFactory

_PROVIDERS: Dict[str, ProviderFactory] = {}
_PROVIDER_ALIASES = {
    "google_web": "googletrans",
    "google_translate": "googletrans",
    "google_cloud": "google_cloud",
    "gcloud": "google_cloud",
    "alibaba": "alibaba",
    "yandex": "yandex",
    "openai_compatible": "openai_compatible",
}


def register_provider(
    name: str,
) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _PROVIDERS[name] = factory
        return factory

    return decorator


def list_providers() -> List[str]:
    return sorted(_PROVIDERS.keys())


def resolve_provider_name(name: str) -> str:
    normalized = name.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def get_provider_factory(name: str) -> Optional[ProviderFactory]:
    return _PROVIDERS.get(name)
