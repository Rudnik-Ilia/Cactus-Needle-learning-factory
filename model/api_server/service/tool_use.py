import json
from threading import Lock

from fastapi import HTTPException

from src.core.general_functions import generate_response

from ..schemas import RouteRequest, RouteResponse, ToolCall

inference_lock = Lock()


def tool_use(request: RouteRequest) -> RouteResponse:
    try:
        with inference_lock:
            result: str = generate_response(request)
        if isinstance(result, str):
            result = json.loads(result)
        tool_calls = [ToolCall.model_validate(call) for call in result]

        return RouteResponse(tool_calls=tool_calls)

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Needle LLM returned an invalid response: {exc}") from exc
