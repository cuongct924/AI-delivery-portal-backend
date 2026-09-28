"""Synthetic "ask_user" tool — not backed by any MCP server. Lets
routers/portal_assistant.py's model surface a clarifying question as
structured, clickable options instead of prose the user has to retype
from verbatim. The mlops persona prompt already tells the model to ask
before guessing a business-defining field with real `oneOf`/`enum`
options; this just gives the drawer something to render as chips for
that case instead of plain text.

Not a real MCP tool — tool_loop.py checks a call's name against
`local_tools` and short-circuits before ever reaching
`McpToolRegistry.call_tool` (which would raise `ValueError: Unknown
tool` for a name no connected server owns).
"""

from typing import Final

from mcp_client import ToolSchema

ASK_USER_TOOL_NAME: Final[str] = "ask_user"

ASK_USER_TOOL: Final[ToolSchema] = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL_NAME,
        "description": (
            "Ask the user a single clarifying question with a fixed set of "
            "choices — use this when a Golden Path field's schema "
            "constrains the answer to specific options (a `oneOf`/`enum`), "
            "instead of listing them in prose. The drawer renders "
            "`options` as clickable chips the user picks from, rather "
            "than free text they'd have to retype by hand. Only for "
            "fields with real, fixed options; for a free-text field (a "
            "name, a URI, ...) just ask in your normal reply instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to show the user.",
                },
                "options": {
                    "type": "array",
                    "description": (
                        "2 or more choices, each the field's schema "
                        "`const` (value) and `title` (label)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The option's schema `const`.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Human-readable `title` to show.",
                            },
                        },
                        "required": ["value", "label"],
                    },
                },
            },
            "required": ["question", "options"],
        },
    },
}
