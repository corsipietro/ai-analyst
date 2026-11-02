"""Tests of the graph pieces that need no model: system text, node names, per-node measures."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from bench.run_bench import model_record, tool_record
from sim_analyst import graph
from sim_analyst.graph import DATA_COLLECTED, build_system_text, make_graph


def test_system_text_ends_with_the_output_language():
    """The answer language is a run setting added after the method, context and database description."""
    text = build_system_text("METHOD", "CONTEXT", "REFERENCE", "English")
    assert "## Output language" in text
    assert text.rstrip().endswith("Always answer in English, whatever the language of the question.")


class FakeModel:
    """A chat model that returns scripted replies, ignoring its input."""

    def __init__(self, replies):
        """Keep the replies to give, in order."""
        self.replies = list(replies)

    def bind_tools(self, tools):
        """Tools are ignored."""
        return self

    def invoke(self, messages):
        """Return the next scripted reply."""
        return self.replies.pop(0)


def test_nodes_write_their_name_in_the_metadata(monkeypatch):
    """Explorer replies (also the one replaced by DATA COLLECTED) and the final answer carry their node."""
    scripted = {"explorer": [AIMessage(content="my draft", response_metadata={"eval_count": 5})],
                "interpreter": [AIMessage(content="final", response_metadata={"eval_count": 7})]}
    monkeypatch.setattr(graph, "make_chat_model", lambda provider, role, seed: FakeModel(scripted[role["name"]]))
    settings = {"provider": "x", "limits": {}, "roles": {"explorer": {"name": "explorer"},
                                                         "interpreter": {"name": "interpreter"}}}
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=[],
                              roles={"explore": "go {DATA_COLLECTED}", "interpret": "write"})
    out = make_graph(analyst, seed=1).invoke({"messages": [HumanMessage(content="q")]})
    ai = [m for m in out["messages"] if m.type == "ai"]
    assert [m.content for m in ai] == [DATA_COLLECTED, "final"]
    assert [m.response_metadata["node"] for m in ai] == ["explore", "interpret"]
    assert ai[0].response_metadata["eval_count"] == 5


def ai(node, read, written, read_s, write_s, tool_calls=()):
    """A fake model message with the metadata Ollama would give."""
    meta = {"node": node, "prompt_eval_count": read, "eval_count": written,
            "prompt_eval_duration": int(read_s * 1e9), "eval_duration": int(write_s * 1e9)}
    return AIMessage(content="text", tool_calls=list(tool_calls), response_metadata=meta)


def test_model_record_has_measures_per_node():
    """Calls, tokens and seconds are added up separately for each node."""
    messages = [HumanMessage(content="q"),
                ai("explore", 100, 10, 1.0, 2.0), ai("explore", 150, 20, 2.0, 3.0),
                ai("interpret", 400, 300, 4.0, 50.0)]
    record = model_record(messages)
    assert record["per_node"]["explore"] == {"model_calls": 2, "tokens_read": 250, "tokens_written": 30,
                                             "seconds_reading": 3.0, "seconds_writing": 5.0}
    assert record["per_node"]["interpret"] == {"model_calls": 1, "tokens_read": 400, "tokens_written": 300,
                                               "seconds_reading": 4.0, "seconds_writing": 50.0}
    assert record["model_calls"] == 3 and record["tokens_written"] == 330


def test_tool_calls_belong_to_the_node_that_asked():
    """A tool message counts for the node of the AI message whose call it answers."""
    from langchain_core.messages import ToolMessage
    call = lambda i: {"name": "summarize_results", "args": {}, "id": i}
    messages = [ai("explore", 1, 1, 0, 0, [call("a"), call("b")]),
                ToolMessage(content="{}", name="summarize_results", tool_call_id="a"),
                ToolMessage(content="{}", name="summarize_results", tool_call_id="b"),
                ai("interpret", 1, 1, 0, 0, [call("c")]),
                ToolMessage(content="{}", name="summarize_results", tool_call_id="c")]
    assert tool_record(messages)["tool_calls_per_node"] == {"explore": 2, "interpret": 1}


def meta(read, written, read_s=1.0, write_s=2.0):
    """Ollama-like metadata of one call."""
    return {"prompt_eval_count": read, "eval_count": written,
            "prompt_eval_duration": int(read_s * 1e9), "eval_duration": int(write_s * 1e9)}


def run_graph(monkeypatch, explorer_replies, interpreter_replies):
    """Run the graph with scripted replies (no tools); return the messages."""
    scripted = {"explorer": list(explorer_replies), "interpreter": list(interpreter_replies)}
    monkeypatch.setattr(graph, "make_chat_model", lambda provider, role, seed: FakeModel(scripted[role["name"]]))
    settings = {"provider": "x", "limits": {"empty_retries": 1, "handoffs": 1},
                "roles": {"explorer": {"name": "explorer"}, "interpreter": {"name": "interpreter"}}}
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=[],
                              roles={"explore": "go {DATA_COLLECTED}", "interpret": "write"})
    return make_graph(analyst, seed=1).invoke({"messages": [HumanMessage(content="q")]})["messages"]


def test_empty_reply_then_full_reply_counts_both_calls(monkeypatch):
    """The discarded empty reply is saved in the kept reply's metadata and counted by the bench."""
    messages = run_graph(monkeypatch,
                         [AIMessage(content="", response_metadata=meta(100, 5))],
                         [AIMessage(content="", response_metadata=meta(400, 300, 4.0, 50.0)),
                          AIMessage(content="final", response_metadata=meta(410, 20, 4.0, 5.0))])
    final = messages[-1]
    assert final.content == "final"
    assert final.response_metadata["discarded_calls"] == [
        {"node": "interpret", "reason": "empty_retry", "prompt_eval_count": 400, "eval_count": 300,
         "prompt_eval_duration": 4_000_000_000, "eval_duration": 50_000_000_000}]
    record = model_record(messages)
    assert record["empty_retries"] == 1 and record["handoffs"] == 0
    assert record["model_calls"] == 3 and record["tokens_written"] == 5 + 300 + 20
    assert record["seconds_writing"] == 2.0 + 50.0 + 5.0
    assert record["seconds_reading"] == 1.0 + 4.0 + 4.0
    assert record["per_node"]["interpret"]["model_calls"] == 2
    assert record["per_node"]["interpret"]["tokens_read"] == 810
    assert record["per_node"]["interpret"]["seconds_writing"] == 55.0


def test_empty_replies_before_a_handoff_are_counted(monkeypatch):
    """Both empty replies (first call and retry) are saved in the handoff message and counted."""
    empty = lambda: AIMessage(content="", response_metadata=meta(300, 200, 3.0, 30.0))
    messages = run_graph(monkeypatch,
                         [AIMessage(content="", response_metadata=meta(100, 5)),
                          AIMessage(content="", response_metadata=meta(120, 6))],
                         [empty(), empty(), AIMessage(content="final", response_metadata=meta(500, 40, 5.0, 8.0))])
    handoff = next(m for m in messages if m.type == "human" and m.content.startswith(graph.HANDOFF_MARK))
    assert [c["reason"] for c in handoff.additional_kwargs["discarded_calls"]] == ["empty_retry", "handoff"]
    record = model_record(messages)
    assert record["empty_retries"] == 1 and record["handoffs"] == 1
    assert record["model_calls"] == 2 + 2 + 1
    assert record["tokens_written"] == 5 + 6 + 400 + 40
    assert record["per_node"]["interpret"]["model_calls"] == 3
    assert record["per_node"]["explore"]["model_calls"] == 2
    assert record["per_node"]["interpret"]["tokens_read"] == 300 + 300 + 500


def test_calculate_is_offered_to_the_graph():
    """The graph gives the model the calculator, together with the data tools."""
    assert "calculate" in graph.GRAPH_TOOLS
