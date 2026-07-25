import json
from pathlib import Path

import needle
from prompt_toolkit import prompt

from ..settings import SETTINGS


def load_tools(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data)


def main(query: str) -> None:
    params, config = needle.load_checkpoint(SETTINGS.path_to_model)
    model = needle.SimpleAttentionNetwork(config)
    tokenizer = needle.get_tokenizer()

    result = needle.generate(
        model,
        params,
        tokenizer,
        query=query,
        tools=load_tools(SETTINGS.path_to_tools_file),
        stream=False,
    )
    # breakpoint()
    print(result)


def run() -> None:
    while query := prompt("...> "):
        if query.startswith("/"):
            match query[1:]:
                case "q" | "quit" | "exit":
                    break

            continue

        main(query)


if __name__ == "__main__":
    run()
