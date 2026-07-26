from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import logging
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from prompt_toolkit import prompt

from agent.config import AgentSettings, build_chat_model
from agent.text_data import HELP_TEXT
from agent.tools import build_tools, handle_tool_errors

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Small wrapper that keeps memory/context control outside LangChain internals."""
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.tools = build_tools()
        self.checkpointer = InMemorySaver()
        self._agent: Any | None = None

    def _render_system_prompt(self) -> str:
        if not self.settings.runtime_context.strip():
            return self.settings.system_prompt

        return (f"""
            {self.settings.system_prompt.rstrip()}\n\n
            Operator-controlled runtime context:\n
            {self.settings.runtime_context.strip()}\n""")

    def _build_agent(self) -> Any:
        return create_agent(
            model=build_chat_model(self.settings),
            tools=self.tools,
            system_prompt=self._render_system_prompt(),
            checkpointer=self.checkpointer,
            middleware=[handle_tool_errors],
        )

    def _invoke_config(self) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": self.settings.thread_id},
            "recursion_limit": self.settings.recursion_limit,
        }

    def rebuild(self) -> None:
        """Mark the graph dirty after changing prompt, context, model, adapter, or tools."""
        self._agent = None

    def _ensure_agent(self) -> Any:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def reset_all_memory(self) -> None:
        """Reset in-memory checkpoints for all threads in this process."""

        self.checkpointer = InMemorySaver()
        self.rebuild()

    def ask(self, user_text: str) -> str:
        result = self._ensure_agent().invoke(
            {
                "messages": [
                    # {
                    #     "role": "system",
                    #     "content": "Use the current thread memory and available tools when helpful.",
                    # },
                    # {
                    #     "role": "assistant",
                    #     "content": "Ready. I will answer using the provided context and tools.",
                    # },
                    {"role": "user", "content": user_text},
                ]
            },
            config=self._invoke_config(),
        )
        last_message = result["messages"][-1]
        content = getattr(last_message, "content", last_message)
        return str(content)

    def set_thread(self, thread_id: str) -> None:
        self.settings.thread_id = thread_id

    def set_model(self, model: str) -> None:
        self.settings.model = model
        self.rebuild()

    def set_temperature(self, temperature: float) -> None:
        self.settings.temperature = temperature
        self.rebuild()

    def set_runtime_context(self, context: str) -> None:
        self.settings.runtime_context = context
        self.rebuild()

    def add_runtime_context(self, context: str) -> None:
        current = self.settings.runtime_context.strip()
        next_context = context.strip()
        self.settings.runtime_context = f"{current}\n{next_context}".strip()
        self.rebuild()

    def _memory_config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self.settings.thread_id}}

    # def describe(self) -> str:
    #     tool_names = ", ".join(tool.name for tool in self.tools) or "none"
    #     base_url = self.settings.base_url
    #     return (
    #         f"adapter={self.settings.llm_adapter}\n"
    #         f"model={self.settings.model}\n"
    #         f"base_url={base_url or 'default provider'}\n"
    #         f"api_key={self.settings.api_key}\n"
    #         f"temperature={self.settings.temperature}\n"
    #         f"num_ctx={self.settings.num_ctx or 'adapter default'}\n"
    #         f"thread_id={self.settings.thread_id}\n"
    #         f"recursion_limit={self.settings.recursion_limit}\n"
    #         f"tools={tool_names}\n"
    #         f"runtime_context={'set' if self.settings.runtime_context.strip() else 'empty'}"
    #     )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft LangChain CLI agent")
    parser.add_argument("--adapter", help="LLM adapter: langchain, ollama, llama-cpp")
    parser.add_argument("--model", help="Model name, e.g. openai:gpt-5.5, llama3.1, local-model")
    parser.add_argument("--base-url", help="Adapter base URL")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible adapters")
    parser.add_argument("--temperature", type=float, help="Model temperature")
    parser.add_argument("--num-ctx", type=int, help="Context window override for supported local adapters")
    parser.add_argument("--thread-id", help="Memory thread id")
    parser.add_argument("--recursion-limit", type=int, help="Max agent graph steps per turn")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def new_thread_id() -> str:
    return datetime.now(ZoneInfo("UTC")).strftime("thread-%Y%m%d-%H%M%S")


def handle_command(agent: GeneralAgent, raw: str) -> bool:
    command, _, rest = raw.partition(" ")
    rest = rest.strip()
    match command:
        case "/q":
            return False

        case "/help":
            print(HELP_TEXT)
            return True

        case "/tools":
            for registered_tool in agent.tools:
                print(f"- {registered_tool.name}: {registered_tool.description}")
            return True

        case "/model":
            if not rest:
                print(agent.settings.model)
            else:
                agent.set_model(rest)
                print(f"Model set to {agent.settings.model}")
            return True

        case "/temperature":
            if not rest:
                print(agent.settings.temperature)
            else:
                try:
                    agent.set_temperature(float(rest))
                except ValueError:
                    print("Usage: /temperature <float>")
                    return True
                print(f"Temperature set to {agent.settings.temperature}")
            return True

        case "/thread":
            if rest:
                agent.set_thread(rest)
                print(f"Switched thread_id to {agent.settings.thread_id}")
            else:
                print(agent.settings.thread_id)
            return True

        case "/new":
            agent.set_thread(rest or new_thread_id())
            print(f"Started thread_id {agent.settings.thread_id}")
            return True

        case "/reset-memory":
            agent.reset_all_memory()
            print("Cleared all in-process memory checkpoints.")
            return True

        case "/ctx":
            subcommand, _, value = rest.partition(" ")
            value = value.strip()
            if subcommand == "show":
                print(agent.settings.runtime_context or "<empty>")
            elif subcommand == "set" and value:
                agent.set_runtime_context(value)
                print("Runtime context replaced.")
            elif subcommand == "add" and value:
                agent.add_runtime_context(value)
                print("Runtime context appended.")
            elif subcommand == "clear":
                agent.set_runtime_context("")
                print("Runtime context cleared.")
            else:
                print("Usage: /context show | set <text> | add <text> | clear")
            return True


    print(f"Unknown command: {command}. Type /help.")
    return True


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    agent = GeneralAgent(AgentSettings())
    run_cli(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
