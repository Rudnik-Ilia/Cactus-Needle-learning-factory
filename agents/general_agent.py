from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agents.config import AgentSettings, build_chat_model
from agents.tools import build_tools, handle_tool_errors


class GeneralAgent:
    """Small wrapper that keeps memory/context control outside LangChain internals."""
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.tools = build_tools()
        self.checkpointer = InMemorySaver()
        self._agent: CompiledStateGraph | None = None

    def _render_system_prompt(self) -> str:
        system_prompt = self.settings.system_prompt.rstrip()
        runtime_context = self.settings.runtime_context.strip()

        if not runtime_context:
            return system_prompt

        return (
            f"{system_prompt}\n\n"
            "Operator-controlled runtime context:\n"
            f"{runtime_context}"
        )

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
        """Mark the graph dirty after changing prompt, context, model, or tools."""
        self._agent = None

    def _ensure_agent(self) -> Any:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def reset_all_memory(self) -> None:
        """Reset in-memory checkpoints for all threads in this process."""
        self.checkpointer = InMemorySaver()
        self.rebuild()

    def list_tools(self) -> list[Any]:
        return list(self.tools)

    def current_thread_id(self) -> str:
        return self.settings.thread_id

    def current_runtime_context(self) -> str:
        return self.settings.runtime_context

    def status_lines(self) -> list[str]:
        tool_names = ", ".join(tool.name for tool in self.tools) or "<none>"
        context_status = "set" if self.settings.runtime_context.strip() else "empty"
        return [
            f"model: {self.settings.model}",
            f"thread_id: {self.settings.thread_id}",
            f"tools: {tool_names}",
            f"runtime_context: {context_status}",
        ]

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
        last_message: str = result["messages"][-1]
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
