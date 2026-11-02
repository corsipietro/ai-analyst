"""Build the chat model of one role from the settings: the only place that knows the provider.

To add a provider (e.g. an API), add a branch here; the graph only sees a LangChain chat model.
"""
from __future__ import annotations


def make_chat_model(provider: str, role: dict, seed: int):
    """Return a LangChain chat model for one role (model, reasoning, sampling, token cap) and seed."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        sampling = {k: role[k] for k in ("temperature", "top_k", "top_p") if k in role}
        return ChatOllama(model=role["model"], reasoning=role.get("reasoning", False),
                          num_predict=role.get("max_tokens"), seed=seed, **sampling)
    raise NotImplementedError(f"provider {provider!r} is not implemented: see sim_analyst/llm.py.")
