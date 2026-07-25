from typing import Any

from pydantic import BaseModel
from pydantic.fields import Field


class ToolParameter(BaseModel):
    type: str
    description: str = ""
    required: bool = False


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, ToolParameter] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    query: str = Field(min_length=11)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    tool_calls: list[ToolCall]
