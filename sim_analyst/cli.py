"""Command line: ask one question, or print the database description the analyst receives.

    python -m sim_analyst describe --domain <name>
    python -m sim_analyst ask --domain <name> "What does the experiment show?"
"""
from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage


def main() -> None:
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(prog="sim_analyst", description="Ask questions about simulation results.")
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe", help="print the database description given to the analyst")
    describe.add_argument("--domain", required=True)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("--domain", required=True)
    ask.add_argument("--seed", type=int, default=1)
    ask.add_argument("--show-steps", action="store_true", help="print every message, not only the answer")
    ask.add_argument("question")
    args = parser.parse_args()

    from .graph import open_analyst
    from .graphs import build_graph
    analyst = open_analyst(args.domain)
    try:
        if args.command == "describe":
            print(json.dumps(json.loads(analyst.reference), indent=2, ensure_ascii=False))
            return
        graph = build_graph(analyst, seed=args.seed)
        limit = analyst.domain.settings["limits"]["recursion_limit"]
        state = None
        for state in graph.stream({"messages": [HumanMessage(content=args.question)]},
                                  {"recursion_limit": limit}, stream_mode="values"):
            if args.show_steps:
                state["messages"][-1].pretty_print()
        if state and not args.show_steps:
            print(state["messages"][-1].content)
    finally:
        analyst.close()
