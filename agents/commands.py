from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TextIO
from zoneinfo import ZoneInfo


class RegisteredTool(Protocol):
    name: str
    description: str


class AgentSession(Protocol):
    def ask(self, user_text: str) -> str: ...

    def list_tools(self) -> Iterable[RegisteredTool]: ...

    def current_thread_id(self) -> str: ...

    def set_thread(self, thread_id: str) -> None: ...

    def reset_all_memory(self) -> None: ...

    def current_runtime_context(self) -> str: ...

    def set_runtime_context(self, context: str) -> None: ...

    def add_runtime_context(self, context: str) -> None: ...

    def status_lines(self) -> Iterable[str]: ...


CommandHandler = Callable[[AgentSession, str, TextIO], bool]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    summary: str
    handler: CommandHandler
    group: str = "Commands"
    aliases: tuple[str, ...] = ()
    usage: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def new_thread_id() -> str:
    return datetime.now(ZoneInfo("UTC")).strftime("thread-%Y%m%d-%H%M%S")


def render_help(commands: Iterable[CommandSpec] | None = None) -> str:
    grouped: dict[str, list[CommandSpec]] = defaultdict(list)
    for command in commands or COMMANDS:
        grouped[command.group].append(command)

    lines: list[str] = []
    for group, specs in grouped.items():
        if lines:
            lines.append("")
        lines.append(f"{group}:")
        for spec in specs:
            names = ", ".join(spec.names)
            lines.append(f"  {names:<28} {spec.summary}")
            for usage in spec.usage:
                lines.append(f"  {usage:<28}")
    lines.append("")
    lines.append("Any other input is sent to the agent.")
    return "\n".join(lines)


def dispatch_command(agent: AgentSession, raw: str, output: TextIO | None = None) -> bool:
    stream: TextIO = output if output is not None else sys.stdout
    command, _, rest = raw.partition(" ")
    spec = COMMAND_BY_NAME.get(command)
    if spec is None:
        print(f"Unknown command: {command}. Type /help.", file=stream)
        return True

    return spec.handler(agent, rest.strip(), stream)


def _quit(_agent: AgentSession, _rest: str, _output: TextIO) -> bool:
    return False


def _help(_agent: AgentSession, _rest: str, output: TextIO) -> bool:
    print(render_help(), file=output)
    return True


def _status(agent: AgentSession, _rest: str, output: TextIO) -> bool:
    for line in agent.status_lines():
        print(line, file=output)
    return True


def _tools(agent: AgentSession, _rest: str, output: TextIO) -> bool:
    for registered_tool in agent.list_tools():
        print(f"- {registered_tool.name}: {registered_tool.description}", file=output)
    return True


def _thread(agent: AgentSession, rest: str, output: TextIO) -> bool:
    if rest:
        agent.set_thread(rest)
        print(f"Switched thread_id to {agent.current_thread_id()}", file=output)
    else:
        print(agent.current_thread_id(), file=output)
    return True


def _new(agent: AgentSession, rest: str, output: TextIO) -> bool:
    agent.set_thread(rest or new_thread_id())
    print(f"Started thread_id {agent.current_thread_id()}", file=output)
    return True


def _reset_memory(agent: AgentSession, _rest: str, output: TextIO) -> bool:
    agent.reset_all_memory()
    print("Cleared all in-process memory checkpoints.", file=output)
    return True


def _context(agent: AgentSession, rest: str, output: TextIO) -> bool:
    subcommand, _, value = rest.partition(" ")
    value = value.strip()

    match subcommand:
        case "show":
            print(agent.current_runtime_context() or "<empty>", file=output)
        case "set" if value:
            agent.set_runtime_context(value)
            print("Runtime context replaced.", file=output)
        case "add" if value:
            agent.add_runtime_context(value)
            print("Runtime context appended.", file=output)
        case "clear":
            agent.set_runtime_context("")
            print("Runtime context cleared.", file=output)
        case _:
            print("Usage: /context show | set <text> | add <text> | clear", file=output)
    return True


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "Show commands", _help),
    CommandSpec("/q", "Quit", _quit, aliases=("/exit",)),
    CommandSpec("/status", "Show model, thread, tools, and context status", _status),
    CommandSpec("/tools", "List registered tools", _tools),
    CommandSpec("/thread", "Show or switch the memory thread", _thread, group="Memory/thread control", usage=("/thread <id>",)),
    CommandSpec("/new", "Start a new memory thread", _new, group="Memory/thread control", usage=("/new [id]",)),
    CommandSpec("/reset-memory", "Clear all in-process memory checkpoints", _reset_memory, group="Memory/thread control"),
    CommandSpec("/context", "Show or change operator-controlled runtime context", _context, group="Context/prompt control", aliases=("/ctx",), usage=(
            "/context show",
            "/context set <text>",
            "/context add <text>",
            "/context clear",
        ),
    ),
)

COMMAND_BY_NAME = {name: spec for spec in COMMANDS for name in spec.names}
