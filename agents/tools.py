from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


def build_tools() -> list[BaseTool]:
    """Register agent tools in one place.

    Add your project tools here later. Keep each tool small, typed, and with a
    clear docstring so the model knows when to call it.
    """

    return [get_current_datetime]

@wrap_tool_call()
def handle_tool_errors(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
    try:
        return handler(request)
    except Exception as exc:
        logger.exception("Tool call failed: %s", request.tool_call.get("name", "<unknown>"))
        return ToolMessage(content=f"Tool error: {type(exc).__name__}: {exc}", tool_call_id=request.tool_call["id"])

@tool
def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time for an IANA timezone.

    Use this when the user asks about current date/time or when an operation
    needs a timestamp.

    Args:
        timezone_name: IANA timezone name, for example "UTC" or "Europe/London".
    """

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone_name!r}. Use an IANA timezone like 'UTC'."

    return datetime.now(tz).isoformat(timespec="seconds")
