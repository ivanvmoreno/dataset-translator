import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from .base import AsyncTranslator, TranslationResult, normalize_source_lang
from .registry import register_provider


@register_provider("yandex")
def _create_yandex(options: Dict[str, str]) -> AsyncTranslator:
    return YandexTranslate(
        api_key=options.get("api_key"),
        iam_token=options.get("iam_token"),
        folder_id=options.get("folder_id"),
        endpoint=options.get("endpoint"),
        timeout=options.get("timeout"),
    )


class YandexTranslate:

    def __init__(
        self,
        api_key: Optional[str] = None,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self._folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self._endpoint = (
            endpoint or ""
        ).strip() or "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self._timeout = int(timeout) if timeout else 60

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: Optional[str], dest: str
    ) -> List[TranslationResult]:
        self._ensure_api_key()
        self._ensure_folder_id()
        src_lang = normalize_source_lang(src)
        payload: Dict[str, Any] = {
            "targetLanguageCode": dest,
            "texts": texts,
        }
        if self._folder_id:
            payload["folderId"] = self._folder_id
        if src_lang != "auto":
            payload["sourceLanguageCode"] = src_lang

        response = self._post_json(payload)
        translations = response.get("translations", [])
        return [
            TranslationResult(item.get("text", "")) for item in translations
        ]

    def _ensure_api_key(self) -> None:
        if not self._api_key and not self._iam_token:
            raise ValueError(
                "Yandex provider requires api_key, iam_token, or env vars "
                "YANDEX_API_KEY/YANDEX_IAM_TOKEN."
            )

    def _ensure_folder_id(self) -> None:
        if not self._folder_id:
            raise ValueError(
                "Yandex provider requires folder_id or YANDEX_FOLDER_ID."
            )

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        auth_header = (
            f"Api-Key {self._api_key}"
            if self._api_key
            else f"Bearer {self._iam_token}"
        )
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(f"Yandex Translate API error: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Yandex Translate API request failed: {exc}"
            ) from exc
        return json.loads(data)
