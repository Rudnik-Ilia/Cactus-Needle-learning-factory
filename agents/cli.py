import logging

from prompt_toolkit.shortcuts.prompt import prompt

from agents.commands import dispatch_command
from agents.general_agent import GeneralAgent

logger = logging.getLogger(__name__)


def handle_command(agent: GeneralAgent, raw: str) -> bool:
    return dispatch_command(agent, raw)

def run_cli(agent: GeneralAgent) -> None:
    print("General LangChain agent draft. Type /help for commands, /q to quit.")

    while True:
        try:
            raw = prompt("\n you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not raw:
            continue

        if raw.startswith("/"):
            if not handle_command(agent, raw):
                print("bye")
                return
            continue
        try:
            print(f"\nagent> \n {agent.ask(raw)}", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as exc:
            logger.exception("Agent invocation failed")
            print(f"Agent error: {type(exc).__name__}: {exc}")
