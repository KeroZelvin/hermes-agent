import json
from unittest.mock import patch

from tools import web_tools


class TestMiniMaxWebBackend:
    @patch("tools.web_tools._load_web_config", return_value={"backend": "minimax"})
    def test_explicit_minimax_backend_selected(self, _cfg):
        assert web_tools._get_backend() == "minimax"

    @patch("tools.web_tools._load_web_config", return_value={"backend": "minimax"})
    @patch("tools.web_tools._is_minimax_mcp_backend_ready", return_value=True)
    def test_check_web_api_key_uses_minimax_mcp(self, _ready, _cfg):
        assert web_tools.check_web_api_key() is True

    @patch("tools.web_tools._load_web_config", return_value={"backend": "minimax"})
    @patch("tools.web_tools._is_minimax_mcp_backend_ready", return_value=True)
    def test_extract_hidden_for_minimax_search_only_backend(self, _ready, _cfg):
        assert web_tools.check_web_extract_backend() is False

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("tools.web_tools._load_web_config", return_value={"backend": "minimax"})
    @patch("tools.web_tools.registry.dispatch")
    def test_web_search_tool_normalizes_minimax_mcp_results(self, mock_dispatch, _cfg, _interrupt):
        mock_dispatch.return_value = json.dumps(
            {
                "result": json.dumps(
                    {
                        "organic": [
                            {
                                "title": "MiniMax result",
                                "link": "https://example.com/minimax",
                                "snippet": "MiniMax search result",
                                "date": "2026-04-03",
                            }
                        ],
                        "related_searches": [{"query": "MiniMax Hermes MCP"}],
                        "base_resp": {"status_code": 0, "status_msg": "success"},
                    }
                )
            }
        )

        result = json.loads(web_tools.web_search_tool("MiniMax Hermes", limit=3))

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "MiniMax result",
                "url": "https://example.com/minimax",
                "description": "MiniMax search result (2026-04-03)",
                "position": 1,
            }
        ]
        assert result["data"]["related_searches"] == ["MiniMax Hermes MCP"]
        assert result["meta"]["provider"] == "minimax-mcp"

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("tools.web_tools._load_web_config", return_value={"backend": "minimax"})
    @patch("tools.web_tools.registry.dispatch", return_value=json.dumps({"error": "MCP unavailable"}))
    def test_web_search_tool_surfaces_minimax_mcp_errors(self, _dispatch, _cfg, _interrupt):
        result = json.loads(web_tools.web_search_tool("MiniMax Hermes", limit=3))
        assert result["error"] == "Error searching web: MCP unavailable"
