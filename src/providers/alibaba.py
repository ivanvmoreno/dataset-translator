import asyncio
import os
from typing import Any, Dict, List, Optional
from src.rate_limiters import SyncTokenBucketLimiter
from .base import AsyncTranslator, TranslationResult, normalize_source_lang
from .registry import register_provider


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


class AlibabaTranslate:

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
                self._tpm_limiter.acquire(
                    token_count * 2
                )  # input + output (approx. translation length)
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
                num_retries=0,
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
            raise ValueError("tokens_per_minute must be an integer.") from exc
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
