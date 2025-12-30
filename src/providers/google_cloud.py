import asyncio
from typing import Dict, List, Optional
from .base import (
    AsyncTranslator,
    TranslationResult,
    normalize_source_lang,
    parse_bool,
)
from .registry import register_provider


@register_provider("google_cloud")
def _create_google_cloud(options: Dict[str, str]) -> AsyncTranslator:
    return GoogleCloudTranslate(
        api_endpoint=options.get("api_endpoint"),
        api_key=options.get("api_key"),
        anonymous=options.get("anonymous") or options.get("no_auth"),
    )


class GoogleCloudTranslate:

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        anonymous: Optional[str] = None,
    ) -> None:
        from google.api_core.client_options import ClientOptions
        from google.cloud import translate_v2 as translate_v2
        from google.auth.credentials import AnonymousCredentials

        self._client_options = None
        if api_endpoint or api_key:
            self._client_options = ClientOptions(
                api_endpoint=api_endpoint, api_key=api_key
            )
        self._translate_v2 = translate_v2
        credentials = AnonymousCredentials() if parse_bool(anonymous) else None
        self.client = translate_v2.Client(
            client_options=self._client_options, credentials=credentials
        )

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: Optional[str], dest: str
    ) -> List[TranslationResult]:
        src_lang = normalize_source_lang(src)
        request = {"target_language": dest, "format_": "text"}
        if src_lang != "auto":
            request["source_language"] = src_lang
        results = self.client.translate(texts, **request)
        if isinstance(texts, str):
            results = [results]
        return [
            TranslationResult(item.get("translatedText", ""))
            for item in results
        ]
