

from dataclasses import dataclass
from typing import Any

from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import SecretStr

from agent.text_data import DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True)
class AgentSettings:
    model: str = "gemma4"
    thread_id: str = "1"
    recursion_limit: int = 10
    llm_adapter: str = "llamacpp"
    base_url: str  = "http://10.100.102.5:8888/v1"
    api_key: str = "No-need"
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: int = 120
    max_retries: int = 3
    num_ctx: int = 30000
    logprobs: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    runtime_context: str = ""
    debug: bool = False


def build_chat_model(settings: AgentSettings) -> Any:
    llm = ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=SecretStr(settings.api_key),
        temperature=settings.temperature,
        top_p=settings.top_p,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        logprobs=settings.logprobs,
    )
    return llm
