
import needle
from prompt_toolkit import prompt

from src.core.general_functions import load_tools
from src.core.settings import SETTINGS


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
