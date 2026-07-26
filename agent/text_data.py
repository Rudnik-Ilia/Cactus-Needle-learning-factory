"""Draft LangChain agent with CLI, memory, context control, tools, and LLM adapters.

Run:
    uv run python agent/general_agent.py

Default remote model:
    OPENAI_API_KEY=...
    uv run python agent/general_agent.py --adapter langchain --model openai:gpt-5.5

Ollama adapter:
    ollama serve
    ollama pull llama3.1
    uv run python agent/general_agent.py --adapter ollama --model llama3.1

llama.cpp adapter through llama-server OpenAI-compatible API:
    llama-server -m ./models/model.gguf --host 127.0.0.1 --port 8080
    uv run python agent/general_agent.py --adapter llama-cpp --model local-model --base-url http://10.100.102.5:8888/v1

Optional environment:
    AGENT_LLM_ADAPTER=langchain|ollama|llama-cpp
    AGENT_MODEL=openai:gpt-5.5
    AGENT_THREAD_ID=local-dev
    AGENT_RECURSION_LIMIT=12
    AGENT_TEMPERATURE=0.2
    AGENT_NUM_CTX=8192
    AGENT_BASE_URL=...
    AGENT_OLLAMA_BASE_URL=http://localhost:11434
    AGENT_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
    AGENT_API_KEY=...
"""

HELP_TEXT = """Commands:
  /help                         Show commands
  /exit                         Quit
  /status                       Show model, thread, tools, and context status
  /adapters                     List available LLM adapters
  /tools                        List registered tools

LLM adapter control:
  /adapter                      Show current adapter
  /adapter <name>               Set adapter: langchain, ollama, llama-cpp
  /model                        Show current model
  /model <name>                 Change model and rebuild agent
  /base-url                     Show current/default base URL
  /base-url <url>               Set base URL for Ollama or llama.cpp
  /base-url clear               Use adapter default base URL
  /api-key show                 Show masked API key status
  /api-key set <key>            Set API key for OpenAI-compatible adapters
  /api-key clear                Clear API key
  /temperature                  Show temperature
  /temperature <float>          Set temperature and rebuild agent
  /num-ctx                      Show context window override
  /num-ctx <int>                Set context window for Ollama
  /num-ctx clear                Use adapter default context window

Memory/thread control:
  /thread                       Show current thread id
  /thread <id>                  Switch to an existing/new memory thread
  /new [id]                     Start a new memory thread; id defaults to timestamp
  /mem                          Show stored messages for current thread
  /mem raw                      Show raw latest checkpoint for current thread
  /reset-memory                 Clear all in-process MemorySaver checkpoints

Context/prompt control:
  /context show                 Show operator-controlled runtime context
  /context set <text>           Replace runtime context
  /context add <text>           Append runtime context
  /context clear                Clear runtime context
  /system show                  Show system prompt
  /system set <text>            Replace system prompt

Any other input is sent to the agent.
"""

DEFAULT_SYSTEM_PROMPT: str = """You are a practical production-grade assistant.
Rules:
- Be concise and precise.
- Use tools when they are useful or required.
- If a tool fails, recover when possible and explain what happened.
- Do not invent tool results. If a tool is missing, say what tool is needed.
- Respect the runtime context supplied by the operator.
"""
