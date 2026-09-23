"""agents/mcp-servers/golden-path-guide-server/server.py's cost tool.

Loaded via conftest.load_module_from_path (this server has no sibling
thunder_client.py/token_verifier.py, so load_mcp_server doesn't apply)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import load_module_from_path

_SERVER = Path(__file__).parent.parent / "agents/mcp-servers/golden-path-guide-server/server.py"
server = load_module_from_path("golden_path_guide_server", _SERVER)


def test_estimate_golden_path_cost_posts_and_shapes_the_response() -> None:
    response = MagicMock()
    response.json.return_value = {
        "estimated_cost": 12.5,
        "currency": "USD",
        "breakdown": {"gpu": 12.5},
    }
    with patch.object(server.httpx, "post", return_value=response) as post:
        result = server.estimate_golden_path_cost("llm-serve-deploy", "run", {"gpuType": "H100"})

    assert result["estimated_cost"] == 12.5
    assert result["breakdown"] == {"gpu": 12.5}
    _, kwargs = post.call_args
    assert kwargs["json"]["golden_path"] == "llm-serve-deploy"
    assert kwargs["json"]["params"] == {"gpuType": "H100"}
