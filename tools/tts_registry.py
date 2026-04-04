"""Shared TTS provider registry used by tool, gateway, and live voice paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

SynthesizeFn = Callable[[str, str, Dict[str, Any]], str]
AvailabilityFn = Callable[[Dict[str, Any]], bool]
StreamingSetupFn = Callable[[Dict[str, Any]], Any]
StreamingSentenceFn = Callable[[str, Dict[str, Any], Any, Any], None]
StreamingTeardownFn = Callable[[Any], None]


@dataclass(frozen=True)
class RegisteredTTSProvider:
    synthesize: SynthesizeFn
    is_available: AvailabilityFn
    supports_native_opus: bool = False
    is_streaming_available: Optional[AvailabilityFn] = None
    streaming_setup: Optional[StreamingSetupFn] = None
    stream_sentence: Optional[StreamingSentenceFn] = None
    streaming_teardown: Optional[StreamingTeardownFn] = None


_TTS_PROVIDERS: Dict[str, RegisteredTTSProvider] = {}


def register_tts_provider(
    name: str,
    synthesize: SynthesizeFn,
    *,
    is_available: AvailabilityFn,
    supports_native_opus: bool = False,
    is_streaming_available: Optional[AvailabilityFn] = None,
    streaming_setup: Optional[StreamingSetupFn] = None,
    stream_sentence: Optional[StreamingSentenceFn] = None,
    streaming_teardown: Optional[StreamingTeardownFn] = None,
    override: bool = False,
) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("TTS provider name is required")
    if key in _TTS_PROVIDERS and not override:
        raise ValueError(f"TTS provider already registered: {key}")
    _TTS_PROVIDERS[key] = RegisteredTTSProvider(
        synthesize=synthesize,
        is_available=is_available,
        supports_native_opus=supports_native_opus,
        is_streaming_available=is_streaming_available,
        streaming_setup=streaming_setup,
        stream_sentence=stream_sentence,
        streaming_teardown=streaming_teardown,
    )


def get_tts_provider(name: str) -> RegisteredTTSProvider | None:
    return _TTS_PROVIDERS.get((name or "").strip().lower())


def iter_tts_providers() -> Iterator[Tuple[str, RegisteredTTSProvider]]:
    return iter(_TTS_PROVIDERS.items())
