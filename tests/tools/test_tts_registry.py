"""Tests for the shared TTS provider registry / provider-resolution seam."""

import os
from pathlib import Path
from unittest.mock import patch


def test_resolve_tts_provider_uses_configured_provider_when_available():
    from tools.tts_tool import resolve_tts_provider

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai"}), \
         patch("tools.tts_tool._import_openai_client", return_value=object()), \
         patch("tools.tts_tool._resolve_openai_audio_client_config", return_value=("key", "https://api.openai.com/v1")):
        resolved = resolve_tts_provider()

    assert resolved.requested_provider == "openai"
    assert resolved.provider == "openai"
    assert resolved.available is True
    assert resolved.supports_native_opus is True


def test_resolve_tts_provider_falls_back_from_edge_to_neutts_when_needed():
    from tools.tts_tool import resolve_tts_provider

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}), \
         patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
         patch("tools.tts_tool._check_neutts_available", return_value=True):
        resolved = resolve_tts_provider()

    assert resolved.requested_provider == "edge"
    assert resolved.provider == "neutts"
    assert resolved.available is True


def test_generate_tts_result_uses_registered_provider_and_returns_actual_path(tmp_path):
    from tools.tts_tool import generate_tts_result

    requested_path = tmp_path / "requested.mp3"
    actual_path = tmp_path / "generated.ogg"

    def _fake_generate(text: str, output_path: str, tts_config: dict) -> str:
        Path(actual_path).write_bytes(b"audio")
        return str(actual_path)

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "openai"}), \
         patch("tools.tts_tool.resolve_tts_provider") as mock_resolve:
        from tools.tts_tool import ResolvedTTSProvider
        from tools.tts_registry import RegisteredTTSProvider

        mock_resolve.return_value = ResolvedTTSProvider(
            requested_provider="openai",
            provider="openai",
            provider_entry=RegisteredTTSProvider(
                synthesize=_fake_generate,
                is_available=lambda _cfg: True,
                supports_native_opus=True,
            ),
            available=True,
            error=None,
            supports_native_opus=True,
            tts_config={"provider": "openai"},
        )

        result = generate_tts_result("Hello world", output_path=str(requested_path))

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert result["file_path"] == str(actual_path)
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{actual_path}"
    assert result["voice_compatible"] is True


def test_check_tts_requirements_uses_registry_availability_checks():
    from tools.tts_tool import check_tts_requirements

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}), \
         patch("tools.tts_tool.iter_tts_providers") as mock_iter:
        from tools.tts_registry import RegisteredTTSProvider

        mock_iter.return_value = iter([
            ("edge", RegisteredTTSProvider(synthesize=lambda *_args, **_kwargs: "", is_available=lambda _cfg: False)),
            ("openai", RegisteredTTSProvider(synthesize=lambda *_args, **_kwargs: "", is_available=lambda _cfg: True, supports_native_opus=True)),
        ])

        assert check_tts_requirements() is True


def test_custom_provider_override_survives_resolution(monkeypatch, tmp_path):
    import tools.tts_registry as tts_registry
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_registry, "_TTS_PROVIDERS", {})
    monkeypatch.setattr(tts_tool, "_BUILTIN_TTS_PROVIDERS_REGISTERED", False)

    def _custom_edge(text: str, output_path: str, tts_config: dict) -> str:
        Path(output_path).write_bytes(b"edge")
        return output_path

    tts_registry.register_tts_provider(
        "edge",
        _custom_edge,
        is_available=lambda _cfg: True,
        override=True,
    )

    resolved = tts_tool.resolve_tts_provider({"provider": "edge"})
    assert resolved.provider_entry is not None
    assert resolved.provider_entry.synthesize is _custom_edge

    result = tts_tool.generate_tts_result("hello", output_path=str(tmp_path / "out.mp3"), tts_config={"provider": "edge"})
    assert result["success"] is True
    assert os.path.exists(result["file_path"])
