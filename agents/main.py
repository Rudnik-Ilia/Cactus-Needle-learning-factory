from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import logging

from agents.cli import run_cli
from agents.config import AgentSettings
from agents.general_agent import GeneralAgent

logger = logging.getLogger(__name__)


def main() -> int:
    argparse.ArgumentParser(description="Agent orchestrator").parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    agent = GeneralAgent(AgentSettings())
    run_cli(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
