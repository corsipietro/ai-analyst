"""Choose the graph of a run: settings.graph is "v2" (graph.py), "rewoo" or "rewoo_repair" (rewoo.py)."""
from __future__ import annotations

from .graph import make_graph
from .rewoo import make_rewoo_graph

GRAPHS = {"v2": make_graph, "rewoo": make_rewoo_graph,
          "rewoo_repair": lambda analyst, seed: make_rewoo_graph(analyst, seed, repair=True)}
ANSWER_ROLE = {"v2": "interpreter", "rewoo": "analyst", "rewoo_repair": "analyst"}   # the role that writes the final answer


def graph_name(analyst) -> str:
    """The graph chosen in the run settings (default v2); an unknown name is an error."""
    name = analyst.domain.settings.get("graph", "v2")
    if name not in GRAPHS:
        raise ValueError(f"unknown graph {name!r}; choose one of {sorted(GRAPHS)}")
    return name


def build_graph(analyst, seed: int):
    """Build the graph chosen in the run settings."""
    return GRAPHS[graph_name(analyst)](analyst, seed)
