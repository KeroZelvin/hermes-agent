"""Focused tests for MiniMax TTS batch + websocket support."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


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


def test_get_streaming_tts_backend_supports_minimax_websocket_mode():
    from tools.tts_tool import _get_streaming_tts_backend

    assert _get_streaming_tts_backend({"provider": "minimax", "minimax": {}}) is None
    assert _get_streaming_tts_backend({"provider": "minimax", "minimax": {"streaming_mode": "websocket"}}) == "minimax_websocket"
    assert _get_streaming_tts_backend({"provider": "elevenlabs"}) == "elevenlabs"


def test_get_minimax_ws_url_preserves_prefix_and_strips_v1_suffix():
    from tools.tts_tool import _get_minimax_ws_url

    assert _get_minimax_ws_url("https://api.minimax.io") == "wss://api.minimax.io/ws/v1/t2a_v2"
    assert _get_minimax_ws_url("https://proxy.example.com/minimax/v1") == "wss://proxy.example.com/minimax/ws/v1/t2a_v2"
    assert _get_minimax_ws_url("https://api.minimax.io/v1/t2a_v2") == "wss://api.minimax.io/ws/v1/t2a_v2"


def test_streaming_tts_available_for_minimax_websocket(monkeypatch):
    from tools.tts_tool import streaming_tts_available

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with patch("tools.tts_tool._import_websockets", return_value=SimpleNamespace()), \
         patch("tools.tts_tool._resolve_minimax_api_key", return_value="mm-key"):
        assert streaming_tts_available(
            {"provider": "minimax", "minimax": {"streaming_mode": "websocket", "voice_id": "English_BossyLeader"}},
            validate_setup=True,
        ) is True


def test_stream_minimax_ws_audio_decodes_chunks_and_sends_protocol_messages():
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
                "api_key": "mm-key",
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


def test_stream_minimax_ws_audio_rejects_unexpected_connect_event():
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
                    "api_key": "mm-key",
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


def test_stream_sentence_minimax_ws_cleans_up_ffplay_on_stream_error():
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
                    "api_key": "mm-key",
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


def test_generate_minimax_tts_uses_hermes_env_store_and_normalized_endpoint(tmp_path, monkeypatch):
    from tools.tts_tool import _generate_minimax_tts

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    output_path = tmp_path / "tts.mp3"

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": "616263"},
            }

    with patch("hermes_cli.config.get_env_value", return_value="stored-mm-key"), \
         patch("requests.post", return_value=_FakeResponse()) as mock_post:
        result = _generate_minimax_tts(
            "hello world",
            str(output_path),
            {"minimax": {"base_url": "https://api.minimax.io", "voice_id": "English_BossyLeader"}},
        )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"abc"
    assert mock_post.call_args.args[0] == "https://api.minimax.io/v1/t2a_v2"
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer stored-mm-key"
