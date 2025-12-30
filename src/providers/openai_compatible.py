import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from .base import (
    AsyncTranslator,
    TranslationResult,
    normalize_source_lang,
    parse_bool,
)
from .registry import register_provider


@register_provider("openai_compatible")
def _create_openai_compatible(options: Dict[str, str]) -> AsyncTranslator:
    return OpenAICompatibleTranslate(
        base_url=options.get("base_url"),
        model=options.get("model"),
        api_key=options.get("api_key"),
        timeout=options.get("timeout"),
        temperature=options.get("temperature"),
        max_tokens=options.get("max_tokens"),
        system_prompt=options.get("system_prompt"),
        user_agent=options.get("user_agent"),
        disable_proxy=options.get("disable_proxy"),
        batch=options.get("batch"),
    )


class OpenAICompatibleTranslate:

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[str] = None,
        temperature: Optional[str] = None,
        max_tokens: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_agent: Optional[str] = None,
        disable_proxy: Optional[str] = None,
        batch: Optional[str] = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("VLLM_API_KEY")
        )
        self._base_url = self._normalize_base_url(
            base_url
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
        )
        self._model = (
            model
            or os.getenv("OPENAI_COMPAT_MODEL")
            or os.getenv("VLLM_MODEL")
            or ""
        ).strip()
        if not self._model:
            raise ValueError(
                "OpenAI-compatible provider requires model or "
                "OPENAI_COMPAT_MODEL."
            )
        self._timeout = int(timeout) if timeout else 60
        self._temperature = (
            float(temperature) if temperature is not None else 0.0
        )
        self._max_tokens = int(max_tokens) if max_tokens else None
        self._system_prompt = system_prompt
        self._user_agent = (
            user_agent or os.getenv("OPENAI_COMPAT_USER_AGENT") or None
        )
        self._disable_proxy = parse_bool(
            disable_proxy or os.getenv("OPENAI_COMPAT_DISABLE_PROXY")
        )
        self._batch = parse_bool(batch or os.getenv("OPENAI_COMPAT_BATCH"))

    async def translate(
        self, texts: List[str], src: str, dest: str
    ) -> List[TranslationResult]:
        return await asyncio.to_thread(self._translate_sync, texts, src, dest)

    def _translate_sync(
        self, texts: List[str], src: Optional[str], dest: str
    ) -> List[TranslationResult]:
        src_lang = normalize_source_lang(src)
        if self._batch and texts:
            payload = self._build_payload(
                json.dumps(texts, ensure_ascii=True), src_lang, dest
            )
            try:
                response = self._post_json(payload)
            except RuntimeError:
                return self._translate_individual(texts, src_lang, dest)
            translated_texts = _extract_openai_translated_texts(response)
            if len(translated_texts) != len(texts):
                return self._translate_individual(texts, src_lang, dest)
            return [TranslationResult(text) for text in translated_texts]
        return self._translate_individual(texts, src_lang, dest)

    def _translate_individual(
        self, texts: List[str], src_lang: str, dest: str
    ) -> List[TranslationResult]:
        out: List[TranslationResult] = []
        for text in texts:
            payload = self._build_payload(text, src_lang, dest)
            response = self._post_json(payload)
            translated = _extract_openai_translated_text(response)
            out.append(TranslationResult(translated))
        return out

    def _build_messages(
        self, text: str, src_lang: str, dest: str
    ) -> List[Dict[str, str]]:
        if self._system_prompt:
            prompt = self._render_system_prompt(
                self._system_prompt, src_lang, dest
            )
        elif src_lang == "auto":
            prompt = (
                "Translate to {dest}. Detect the source language. "
                "Return only the translated text."
            ).format(dest=dest)
        else:
            prompt = (
                "Translate from {src} to {dest}. Return only the translated "
                "text."
            ).format(src=src_lang, dest=dest)
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

    def _build_payload(
        self, text: str, src_lang: str, dest: str
    ) -> Dict[str, Any]:
        messages = self._build_messages(text, src_lang, dest)
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        return payload

    def _render_system_prompt(
        self, template: str, src_lang: str, dest: str
    ) -> str:
        try:
            return template.format(src=src_lang, dest=dest)
        except (KeyError, ValueError):
            return template.replace("{src}", src_lang).replace("{dest}", dest)

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._user_agent:
            headers["User-Agent"] = self._user_agent
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            if self._disable_proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({})
                )
                with opener.open(req, timeout=self._timeout) as response:
                    data = response.read().decode("utf-8")
            else:
                with urllib.request.urlopen(
                    req, timeout=self._timeout
                ) as response:
                    data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(
                f"OpenAI-compatible API error: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible API request failed: {exc}"
            ) from exc
        return json.loads(data)

    @staticmethod
    def _normalize_base_url(value: Optional[str]) -> str:
        base = (value or "").strip()
        if not base:
            base = "http://localhost:8000/v1"
        if base.endswith("/"):
            base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base


def _extract_openai_translated_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"OpenAI-compatible API error: {resp['error']}")
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        return ""
    return _extract_openai_choice_text(choices[0])


def _extract_openai_translated_texts(resp: Any) -> List[str]:
    if resp is None:
        return []
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"OpenAI-compatible API error: {resp['error']}")
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        return []
    return [_extract_openai_choice_text(choice) for choice in choices]


def _extract_openai_choice_text(choice: Any) -> str:
    message = choice.get("message") if isinstance(choice, dict) else None
    if message is None:
        message = getattr(choice, "message", None)
    content = None
    if isinstance(message, dict):
        content = message.get("content")
    elif message is not None:
        content = getattr(message, "content", None)
    if content is None and isinstance(choice, dict):
        content = choice.get("text")
    return content if isinstance(content, str) else ""
