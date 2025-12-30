import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol


@dataclass
class TranslationResult:
    text: str


class AsyncTranslator(Protocol):
    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]: ...


ProviderFactory = Callable[[Dict[str, str]], AsyncTranslator]


def normalize_source_lang(source_lang: Optional[str]) -> str:
    if not source_lang:
        return "auto"
    cleaned = source_lang.strip()
    return cleaned or "auto"


def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_provider_options(options: Optional[List[str]]) -> Dict[str, str]:
    if not options:
        return {}
    parsed: Dict[str, str] = {}
    for entry in options:
        if "=" in entry:
            key, value = entry.split("=", 1)
            parsed[key.strip()] = value.strip()
        else:
            parsed[entry.strip()] = "true"
    return parsed
