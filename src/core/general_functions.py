import json
from pathlib import Path

from needle import (
    SimpleAttentionNetwork,
    generate,
    get_tokenizer,
    load_checkpoint,
)

from model.api_server.schemas import RouteRequest

from .settings import SETTINGS

params, config = load_checkpoint(SETTINGS.path_to_model)
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()


def load_tools(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data)


def generate_response(request: RouteRequest) -> str:
    response = generate(
        model,
        params,
        tokenizer,
        query=request.query,
        tools=load_tools(SETTINGS.path_to_tools_file),
        stream=False,
    )
    return response
