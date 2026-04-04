"""Tests for MiniMax TTS and live-streaming integration."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_resolve_minimax_voice_with_explicit_voice_id_skips_voice_list_lookup():
    with patch("tools.tts_tool._get_minimax_system_voices") as mock_voices:
        from tools.tts_tool import _resolve_minimax_voice

        voice_id = _resolve_minimax_voice("key", {"voice_id": "custom-voice-id"}, "https://api.minimax.io")

        assert voice_id == "custom-voice-id"
        mock_voices.assert_not_called()


def test_resolve_minimax_voice_with_legacy_voice_key_skips_voice_list_lookup():
    with patch("tools.tts_tool._get_minimax_system_voices") as mock_voices:
        from tools.tts_tool import _resolve_minimax_voice

        voice_id = _resolve_minimax_voice("key", {"voice": "legacy-voice-id"}, "https://api.minimax.io")

        assert voice_id == "legacy-voice-id"
        mock_voices.assert_not_called()


def test_get_streaming_backend_requires_minimax_websocket_mode():
    from tools.tts_tool import _get_streaming_tts_backend

    assert _get_streaming_tts_backend({"provider": "minimax", "minimax": {}}) is None
    assert _get_streaming_tts_backend({"provider": "minimax", "minimax": {"streaming_mode": "websocket"}}) == "minimax_websocket"
    assert _get_streaming_tts_backend({"provider": "elevenlabs"}) == "elevenlabs"


def test_get_minimax_ws_url_preserves_proxy_prefix_and_strips_v1_suffix():
    from tools.tts_tool import _get_minimax_ws_url

    assert _get_minimax_ws_url("https://api.minimax.io") == "wss://api.minimax.io/ws/v1/t2a_v2"
    assert _get_minimax_ws_url("https://proxy.example.com/minimax/v1") == "wss://proxy.example.com/minimax/ws/v1/t2a_v2"


class _FakeAsyncWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.sent = []

    async def recv(self):
        return next(self._messages)

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class _FakeAsyncConnect:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_minimax_ws_audio_stream_decodes_chunks_and_sends_protocol_messages():
    from tools.tts_tool import _stream_minimax_ws_audio

    fake_ws = _FakeAsyncWebSocket([
        json.dumps({"event": "connected_success", "base_resp": {"status_code": 0, "status_msg": "success"}}),
        json.dumps({"event": "task_started", "base_resp": {"status_code": 0, "status_msg": "success"}}),
        json.dumps({
            "event": "task_continued",
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": "616263"},
            "is_final": True,
        }),
        json.dumps({"event": "task_finished", "base_resp": {"status_code": 0, "status_msg": "success"}}),
    ])
    fake_websockets = SimpleNamespace(connect=lambda *args, **kwargs: _FakeAsyncConnect(fake_ws))
    chunks = []

    with patch("tools.tts_tool._import_websockets", return_value=fake_websockets):
        asyncio.run(_stream_minimax_ws_audio(
            "hello world",
            {
                "api_key": "test-key",
                "base_url": "https://api.minimax.io",
                "model": "speech-2.8-hd",
                "voice_id": "English_BossyLeader",
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            chunks.append,
            threading.Event(),
        ))

    assert chunks == [b"abc"]
    assert [msg["event"] for msg in fake_ws.sent] == ["task_start", "task_continue", "task_finish"]
    assert fake_ws.sent[0]["voice_setting"]["voice_id"] == "English_BossyLeader"
    assert fake_ws.sent[0]["model"] == "speech-2.8-hd"


def test_minimax_ws_audio_stream_rejects_unexpected_connect_event():
    from tools.tts_tool import _stream_minimax_ws_audio

    fake_ws = _FakeAsyncWebSocket([
        json.dumps({"event": "task_started", "base_resp": {"status_code": 0, "status_msg": "success"}}),
    ])
    fake_websockets = SimpleNamespace(connect=lambda *args, **kwargs: _FakeAsyncConnect(fake_ws))

    with patch("tools.tts_tool._import_websockets", return_value=fake_websockets):
        with pytest.raises(ValueError, match="expected connected_success"):
            asyncio.run(_stream_minimax_ws_audio(
                "hello world",
                {
                    "api_key": "test-key",
                    "base_url": "https://api.minimax.io",
                    "model": "speech-2.8-hd",
                    "voice_id": "English_BossyLeader",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                lambda _chunk: None,
                threading.Event(),
            ))


def test_minimax_ws_audio_stream_rejects_unexpected_finish_event():
    from tools.tts_tool import _stream_minimax_ws_audio

    fake_ws = _FakeAsyncWebSocket([
        json.dumps({"event": "connected_success", "base_resp": {"status_code": 0, "status_msg": "success"}}),
        json.dumps({"event": "task_started", "base_resp": {"status_code": 0, "status_msg": "success"}}),
        json.dumps({
            "event": "task_continued",
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": "616263"},
            "is_final": True,
        }),
        json.dumps({"event": "task_continued", "base_resp": {"status_code": 0, "status_msg": "success"}}),
    ])
    fake_websockets = SimpleNamespace(connect=lambda *args, **kwargs: _FakeAsyncConnect(fake_ws))

    with patch("tools.tts_tool._import_websockets", return_value=fake_websockets):
        with pytest.raises(ValueError, match="expected task_finished"):
            asyncio.run(_stream_minimax_ws_audio(
                "hello world",
                {
                    "api_key": "test-key",
                    "base_url": "https://api.minimax.io",
                    "model": "speech-2.8-hd",
                    "voice_id": "English_BossyLeader",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                lambda _chunk: None,
                threading.Event(),
            ))


def test_minimax_ws_stream_sentence_cleans_up_ffplay_on_stream_error():
    from tools.tts_tool import _stream_sentence_minimax_ws

    fake_proc = MagicMock()
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = RuntimeError("wait failed")

    with patch("shutil.which", return_value="/usr/bin/ffplay"), \
         patch("subprocess.Popen", return_value=fake_proc), \
         patch("tools.tts_tool._stream_minimax_ws_audio", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _stream_sentence_minimax_ws(
                "hello world",
                {},
                {
                    "api_key": "test-key",
                    "base_url": "https://api.minimax.io",
                    "model": "speech-2.8-hd",
                    "voice_id": "English_BossyLeader",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                threading.Event(),
            )

    fake_proc.stdin.close.assert_called_once()
    fake_proc.kill.assert_called_once()


def test_play_mp3_bytes_tempfile_closes_handle_before_unlink():
    from tools.tts_tool import _play_mp3_bytes_via_tempfile

    fake_tmp = MagicMock()
    fake_tmp.name = "/tmp/fake-minimax.mp3"

    with patch("tempfile.NamedTemporaryFile", return_value=fake_tmp), \
         patch("builtins.open", MagicMock()), \
         patch("tools.voice_mode.play_audio_file"), \
         patch("os.unlink"):
        _play_mp3_bytes_via_tempfile(b"abc")

    assert fake_tmp.close.call_count >= 1


def test_minimax_tts_success_path_is_successful(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    output_path = tmp_path / "tts.mp3"

    with (
        patch("tools.tts_tool._load_tts_config", return_value={"provider": "minimax", "minimax": {}}),
        patch("tools.tts_tool._resolve_minimax_voice", return_value="voice-1"),
        patch("tools.tts_tool._download_minimax_audio") as mock_download,
        patch("tools.tts_tool._call_minimax_api") as mock_call,
    ):
        mock_call.return_value = {
            "data": {
                "audio": "https://cdn.example.com/audio.mp3"
            },
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        def _write_mp3(url: str, path: str):
            Path(path).write_bytes(b"audio-bytes")

        mock_download.side_effect = _write_mp3

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("Hello world", output_path=str(output_path)))

    assert result["success"] is True
    assert result["provider"] == "minimax"
    assert result["file_path"] == str(output_path)
    assert output_path.exists()

    mock_call.assert_called_once()
    args = mock_call.call_args.args
    assert args[0] == "https://api.minimax.io"
    assert args[1] == "/v1/t2a_v2"
    payload = args[3]

    assert payload["model"] == "speech-2.8-hd"
    assert payload["text"] == "Hello world"
    assert payload["output_format"] == "url"
    assert payload["voice_setting"] == {
        "voice_id": "voice-1",
        "speed": 1.0,
        "vol": 1.0,
        "pitch": 0,
    }
    assert payload["audio_setting"] == {
        "format": "mp3",
        "sample_rate": 32000,
        "bitrate": 128000,
        "channel": 1,
    }


def test_minimax_tts_ogg_output_is_downloaded_to_mp3_then_converted(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    output_path = tmp_path / "custom.ogg"
    download_input = []

    def _write_mp3(url: str, path: str):
        download_input.append(path)
        Path(path).write_bytes(b"audio-bytes")

    def _convert_to_opus(path: str):
        Path(output_path).write_bytes(b"converted-ogg-bytes")
        return str(output_path)

    with (
        patch("tools.tts_tool._load_tts_config", return_value={"provider": "minimax", "minimax": {}}),
        patch("tools.tts_tool._resolve_minimax_voice", return_value="voice-1"),
        patch("tools.tts_tool._convert_to_opus", side_effect=_convert_to_opus) as mock_convert,
        patch("tools.tts_tool._download_minimax_audio") as mock_download,
        patch("tools.tts_tool._call_minimax_api") as mock_call,
    ):
        mock_call.return_value = {
            "data": {
                "audio": "https://cdn.example.com/audio.mp3"
            },
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_download.side_effect = _write_mp3

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("Hello world", output_path=str(output_path)))

    assert result["success"] is True
    assert result["file_path"] == str(output_path)
    assert result["voice_compatible"] is True
    assert len(download_input) == 1
    assert download_input[0].endswith(".mp3")
    mock_convert.assert_called_once_with(download_input[0])


def test_minimax_ogg_output_fails_when_opus_conversion_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    output_path = tmp_path / "custom.ogg"

    with (
        patch("tools.tts_tool._load_tts_config", return_value={"provider": "minimax", "minimax": {}}),
        patch("tools.tts_tool._resolve_minimax_voice", return_value="voice-1"),
        patch("tools.tts_tool._convert_to_opus", return_value=None),
        patch("tools.tts_tool._download_minimax_audio") as mock_download,
        patch("tools.tts_tool._call_minimax_api") as mock_call,
    ):
        mock_call.return_value = {
            "data": {
                "audio": "https://cdn.example.com/audio.mp3"
            },
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        mock_download.side_effect = lambda url, path: Path(path).write_bytes(b"audio-bytes")

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("Hello world", output_path=str(output_path)))

    assert result["success"] is False
    assert "OGG requested but could not be converted" in result["error"]


def test_minimax_rejects_non_default_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    output_path = tmp_path / "tts.mp3"

    with patch("tools.tts_tool._load_tts_config", return_value={
        "provider": "minimax",
        "minimax": {"model": "speech-2.5"},
    }):
        from tools.tts_tool import text_to_speech_tool
        result = json.loads(text_to_speech_tool("Hello world", output_path=str(output_path)))

    assert result["success"] is False
    assert "unsupported for this phase" in result["error"]
    assert "speech-2.8-hd" in result["error"]


def test_minimax_tts_missing_key_is_user_facing_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    output_path = tmp_path / "tts.mp3"

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "minimax", "minimax": {}}):
        from tools.tts_tool import text_to_speech_tool
        result = json.loads(text_to_speech_tool("Hello world", output_path=str(output_path)))

    assert result["success"] is False
    assert "MINIMAX_API_KEY not set" in result["error"]


def test_parse_minimax_status_code_2061_is_reported_as_unsupported_plan_model():
    from tools.tts_tool import _parse_minimax_status_code

    with pytest.raises(ValueError, match="MiniMax unsupported plan/model: status_code=2061"):
        _parse_minimax_status_code(
            {
                "base_resp": {
                    "status_code": 2061,
                    "status_msg": "plan unsupported",
                }
            },
            "MiniMax TTS",
        )


def test_call_minimax_api_non_2xx_returns_provider_error_from_json():
    from tools.tts_tool import _call_minimax_api

    response = SimpleNamespace(
        status_code=429,
        json=lambda: {
            "base_resp": {
                "status_code": 4001,
                "status_msg": "plan quota exceeded",
            }
        },
    )
    response.raise_for_status = MagicMock()

    httpx_client = MagicMock()
    httpx_client.__enter__ = lambda s: httpx_client
    httpx_client.__exit__ = lambda s, exc_type, exc, tb: False
    httpx_client.post.return_value = response

    with patch("tools.tts_tool._import_httpx", return_value=SimpleNamespace(Client=lambda *args, **kwargs: httpx_client)):
        with pytest.raises(ValueError, match="MiniMax MiniMax TTS request failed \\(HTTP 429\\): status_code=4001 plan quota exceeded"):
            _call_minimax_api(
                "https://api.minimax.io",
                "/v1/t2a_v2",
                "token",
                {"text": "hi"},
                "MiniMax TTS",
            )


def test_extract_minimax_audio_url_rejects_malformed_response_envelope():
    from tools.tts_tool import _extract_minimax_audio_url

    with pytest.raises(ValueError, match="response envelope is malformed"):
        _extract_minimax_audio_url({"response": "oops"})
