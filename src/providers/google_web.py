import asyncio
from typing import Any, Dict, List, Optional
from .base import AsyncTranslator, TranslationResult, normalize_source_lang
from .registry import register_provider


@register_provider("googletrans")
def _create_googletrans(options: Dict[str, str]) -> AsyncTranslator:
    return GoogleWebTranslate(proxy=options.get("proxy"))


class GoogleWebTranslate:

    def __init__(self, proxy: Optional[str] = None) -> None:
        from googletrans import Translator

        self.translator = Translator()
        self.proxy = proxy

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        if asyncio.iscoroutinefunction(self.translator.translate):
            results = await self.translator.translate(texts, src=src, dest=dest)
        else:
            results = await asyncio.to_thread(
                self._translate_sync, texts, src, dest
            )

        if not isinstance(results, list):
            results = [results]

        return [TranslationResult(r.text) for r in results]

    def _translate_sync(self, texts: List[str], src: str, dest: str) -> Any:
        src_lang = normalize_source_lang(src)
        return self.translator.translate(texts, src=src_lang, dest=dest)
