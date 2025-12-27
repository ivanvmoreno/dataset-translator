import asyncio
import importlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from .rate_limiters import SyncTokenBucketLimiter

@dataclass
class TranslationResult:
    text: str


class AsyncTranslator(Protocol):
    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]: ...


ProviderFactory = Callable[[Dict[str, str]], AsyncTranslator]


_PROVIDERS: Dict[str, ProviderFactory] = {}
_PROVIDER_ALIASES = {
    "google": "googletrans",
    "google_web": "googletrans",
    "google_cloud": "google_cloud",
    "gcloud": "google_cloud",
    "alibaba": "alibaba",
    "alibabacloud": "alibaba",
    "yandex": "yandex",
}


def normalize_source_lang(source_lang: Optional[str]) -> str:
    if not source_lang:
        return "auto"
    cleaned = source_lang.strip()
    return cleaned or "auto"


def register_provider(
    name: str,
) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _PROVIDERS[name] = factory
        return factory

    return decorator


def list_providers() -> List[str]:
    return sorted(_PROVIDERS.keys())


def _resolve_provider_name(name: str) -> str:
    normalized = name.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


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
    resolved = _resolve_provider_name(provider)
    factory = _PROVIDERS.get(resolved)
    if factory:
        return factory(options)
    if ":" in provider:
        return _load_custom_provider(provider, options)
    available = ", ".join(list_providers())
    raise ValueError(
        f"Unknown provider `{provider}`. Available providers: {available}."
    )


@register_provider("google_cloud")
def _create_google_cloud(options: Dict[str, str]) -> AsyncTranslator:
    return GoogleCloudTranslate(
        api_endpoint=options.get("api_endpoint"),
    )


@register_provider("googletrans")
def _create_googletrans(options: Dict[str, str]) -> AsyncTranslator:
    return GoogleWebTranslate(proxy=options.get("proxy"))


@register_provider("alibaba")
def _create_alibaba(options: Dict[str, str]) -> AsyncTranslator:
    return AlibabaTranslate(
        api_key=options.get("api_key"),
        base_url=options.get("base_url"),
        model=options.get("model"),
        timeout=options.get("timeout"),
        tokens_per_minute=(
            options.get("tokens_per_minute") or options.get("tpm")
        ),
    )


@register_provider("yandex")
def _create_yandex(options: Dict[str, str]) -> AsyncTranslator:
    return YandexTranslate(
        api_key=options.get("api_key"),
        iam_token=options.get("iam_token"),
        folder_id=options.get("folder_id"),
        endpoint=options.get("endpoint"),
        timeout=options.get("timeout"),
    )


class GoogleCloudTranslate:
    """Async wrapper for Google Cloud Translate SDK."""

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
    ) -> None:
        from google.api_core.client_options import ClientOptions
        from google.cloud import translate_v2 as translate_v2

        self._client_options = None
        if api_endpoint:
            self._client_options = ClientOptions(api_endpoint=api_endpoint)
        self._translate_v2 = translate_v2
        self.client = translate_v2.Client(client_options=self._client_options)

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


class GoogleWebTranslate:
    """Async wrapper for the synchronous googletrans library."""

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


class AlibabaTranslate:
    """
    Async wrapper for Alibaba Cloud Model Studio (Qwen-MT) via LiteLLM.

    Credentials:
      - options: api_key
      - OR env:  DASHSCOPE_API_KEY

    Base URL:
      - options: base_url
      - default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
      - Beijing region: https://dashscope.aliyuncs.com/compatible-mode/v1

    Model:
      - options: model (default: qwen-mt-flash)

    Limits:
      - options: tokens_per_minute (or tpm)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[str] = None,
        tokens_per_minute: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._base_url = (
            base_url or ""
        ).strip() or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self._model = (model or "qwen-mt-flash").strip() or "qwen-mt-flash"
        self._timeout = int(timeout) if timeout else 60
        self._tpm_limit = self._parse_positive_int(tokens_per_minute)
        self._tpm_limiter = (
            SyncTokenBucketLimiter(
                self._tpm_limit / 60.0, capacity=self._tpm_limit
            )
            if self._tpm_limit
            else None
        )

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: Optional[str], dest: str
    ) -> List[TranslationResult]:
        self._ensure_api_key()
        src_lang = normalize_source_lang(src)
        out: List[TranslationResult] = []
        request_kwargs: Dict[str, Any] = {}
        if self._timeout:
            request_kwargs["timeout"] = self._timeout

        for text in texts:
            messages = [{"role": "user", "content": text}]
            if self._tpm_limiter:
                token_count = self._estimate_tokens(messages)
                self._tpm_limiter.acquire(token_count)
            response = self._completion(
                messages=messages,
                translation_options={
                    "source_lang": src_lang,
                    "target_lang": dest,
                },
                **request_kwargs,
            )
            translated = _extract_dashscope_translated_text(response)
            out.append(TranslationResult(translated))

        return out

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Alibaba provider requires api_key or DASHSCOPE_API_KEY."
            )

    def _completion(self, **kwargs: Any) -> Any:
        import litellm

        try:
            return litellm.completion(
                model=f"dashscope/{self._model}",
                messages=kwargs.pop("messages"),
                api_key=self._api_key,
                api_base=self._base_url,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Alibaba Model Studio API request failed: {exc}"
            ) from exc

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        import litellm

        try:
            return int(
                litellm.token_counter(
                    model=f"dashscope/{self._model}", messages=messages
                )
            )
        except Exception:
            try:
                from litellm.utils import token_counter

                return int(
                    token_counter(
                        model=f"dashscope/{self._model}", messages=messages
                    )
                )
            except Exception:
                text = " ".join(msg.get("content", "") for msg in messages)
                return max(1, len(text) // 4)

    @staticmethod
    def _parse_positive_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                "tokens_per_minute must be an integer."
            ) from exc
        return parsed if parsed > 0 else None


def _extract_dashscope_translated_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"Alibaba Model Studio API error: {resp['error']}")
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if choices is None:
        choices = getattr(resp, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if message is None:
        message = getattr(first, "message", None)
    if not message:
        return ""
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


class YandexTranslate:
    """
    Async wrapper for Yandex Cloud Translate v2.

    Credentials:
      - options: api_key or iam_token
      - OR env:  YANDEX_API_KEY or YANDEX_IAM_TOKEN
      - when both are set, api_key is used
      - requires ai.translate.user role on the target folder

    Folder:
      - options: folder_id
      - OR env:  YANDEX_FOLDER_ID

    Endpoint:
      - options: endpoint
      - default: https://translate.api.cloud.yandex.net/translate/v2/translate
    """

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
            (endpoint or "").strip()
            or "https://translate.api.cloud.yandex.net/translate/v2/translate"
        )
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
            TranslationResult(item.get("text", ""))
            for item in translations
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
            with urllib.request.urlopen(
                req, timeout=self._timeout
            ) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(
                f"Yandex Translate API error: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Yandex Translate API request failed: {exc}"
            ) from exc
        return json.loads(data)
