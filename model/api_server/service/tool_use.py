

import json
from pathlib import Path
from threading import Lock

from fastapi import HTTPException
from needle import (
    SimpleAttentionNetwork,
    generate,
    get_tokenizer,
    load_checkpoint,
)

from ...settings import SETTINGS
from ..schemas import RouteRequest, RouteResponse, ToolCall

params, config = load_checkpoint(SETTINGS.path_to_model)
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()

inference_lock = Lock()


def load_tools(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data)

def tool_use(request: RouteRequest) -> RouteResponse:
    try:
        with inference_lock:
            result = generate(
                model,
                params,
                tokenizer,
                query=request.query,
                tools=load_tools(SETTINGS.path_to_tools_file),
                stream=False,
            )

        if isinstance(result, str):
            result = json.loads(result)

        tool_calls = [ToolCall.model_validate(call) for call in result]

        return RouteResponse(tool_calls=tool_calls)

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Needle LLM returned an invalid response: {exc}") from exc
