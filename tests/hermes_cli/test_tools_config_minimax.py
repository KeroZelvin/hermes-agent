from unittest.mock import patch

from hermes_cli.tools_config import _run_post_setup


class TestMiniMaxWebPostSetup:
    def test_post_setup_configures_minimax_mcp_server(self):
        config = {}

        with patch("shutil.which", return_value="/usr/bin/uvx"):
            _run_post_setup("minimax_mcp_web", config)

        assert config["web"]["backend"] == "minimax"
        assert config["mcp_servers"]["minimax"] == {
            "command": "/usr/bin/uvx",
            "args": ["minimax-coding-plan-mcp", "-y"],
            "connect_timeout": 120,
            "timeout": 120,
            "env": {
                "MINIMAX_API_KEY": "${MINIMAX_API_KEY}",
                "MINIMAX_API_HOST": "https://api.minimax.io",
            },
            "tools": {
                "include": ["web_search"],
            },
        }

    def test_post_setup_preserves_existing_command_and_include_entries(self):
        config = {
            "mcp_servers": {
                "minimax": {
                    "command": "/custom/uvx",
                    "tools": {"include": ["understand_image"]},
                    "env": {"MINIMAX_API_HOST": "https://custom.minimax.local"},
                }
            }
        }

        with patch("shutil.which", return_value="/usr/bin/uvx"):
            _run_post_setup("minimax_mcp_web", config)

        server = config["mcp_servers"]["minimax"]
        assert server["command"] == "/custom/uvx"
        assert server["tools"]["include"] == ["understand_image", "web_search"]
        assert server["env"]["MINIMAX_API_HOST"] == "https://custom.minimax.local"
        assert server["env"]["MINIMAX_API_KEY"] == "${MINIMAX_API_KEY}"
