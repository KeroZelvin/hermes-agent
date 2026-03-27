from gateway.config import Platform
from gateway.session import SessionSource
import gateway.run as gateway_run


def test_resolve_workspace_cwd_uses_discord_channel_binding(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_gateway_yaml_config",
        {
            "discord": {
                "channel_workspaces": {
                    "123": "/tmp/project-alpha",
                    "123:456": "/tmp/project-thread",
                }
            }
        },
    )
    monkeypatch.setenv("MESSAGING_CWD", "/tmp/default-workspace")

    source = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="channel")
    assert gateway_run._resolve_workspace_cwd(source) == "/tmp/project-alpha"

    threaded = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="thread",
        thread_id="456",
    )
    assert gateway_run._resolve_workspace_cwd(threaded) == "/tmp/project-thread"


def test_resolve_workspace_cwd_falls_back_for_unbound_sources(monkeypatch):
    monkeypatch.setattr(gateway_run, "_gateway_yaml_config", {"discord": {"channel_workspaces": {}}})
    monkeypatch.setenv("MESSAGING_CWD", "/tmp/default-workspace")

    source = SessionSource(platform=Platform.DISCORD, chat_id="999", chat_type="channel")
    assert gateway_run._resolve_workspace_cwd(source) == "/tmp/default-workspace"
