"""Tests of the graph pieces that need no model: system text, node names, per-node measures."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bench.run_bench import model_record, tool_record
from sim_analyst import graph
from sim_analyst.graph import (ANSWER_TOOLS, DATA_COLLECTED, NO_DATA_HEADER, build_system_text, data_retrieved, make_graph,
                              step_error_text)
from sim_analyst.tool_schemas import make_langchain_tools


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
    assert [m.content for m in ai] == [DATA_COLLECTED, NO_DATA_HEADER + "final"]
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
    assert final.content == NO_DATA_HEADER + "final"
    assert final.response_metadata["discarded_calls"] == [
        {"node": "interpret", "reason": "empty_retry", "prompt_eval_count": 400, "eval_count": 300,
         "prompt_eval_duration": 4_000_000_000, "eval_duration": 50_000_000_000,
         "done_reason": "", "reasoning": False}]
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


# ---------------------------------------------------------------- g5.1-a: empty_retry_mode

class RecordingModel(FakeModel):
    """A FakeModel that remembers the messages of each call."""

    def __init__(self, replies):
        """Keep the replies and start an empty record of the calls."""
        super().__init__(replies)
        self.seen = []

    def invoke(self, messages):
        """Record the messages, return the next scripted reply."""
        self.seen.append(messages)
        return super().invoke(messages)


def run_mode(monkeypatch, mode, explorer_replies, first, second, handoffs=1):
    """Run the graph with two interpreter models (reasoning on, then off); return (messages, roles, models)."""
    roles_seen, models = [], {"explorer": FakeModel(explorer_replies), "on": RecordingModel(first),
                              "off": RecordingModel(second)}

    def fake_make(provider, role, seed):
        """Hand out the scripted model of the role and record the role request."""
        roles_seen.append(role)
        if role["name"] == "explorer":
            return models["explorer"]
        return models["on" if role.get("reasoning") else "off"]

    monkeypatch.setattr(graph, "make_chat_model", fake_make)
    limits = {"empty_retries": 1, "handoffs": handoffs, **({"empty_retry_mode": mode} if mode else {})}
    settings = {"provider": "x", "limits": limits,
                "roles": {"explorer": {"name": "explorer"}, "interpreter": {"name": "interpreter", "reasoning": True}}}
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=[],
                              roles={"explore": "go {DATA_COLLECTED}", "interpret": "write", "empty_retry": "again"})
    out = make_graph(analyst, seed=1).invoke({"messages": [HumanMessage(content="q")]})["messages"]
    return out, roles_seen, models


def empty_with(reasoning, **kw):
    """An empty interpreter reply with reasoning text (None: no reasoning_content at all)."""
    extra = {} if reasoning is None else {"reasoning_content": reasoning}
    return AIMessage(content="", additional_kwargs=extra, response_metadata=meta(300, 200, 3.0, 30.0) | kw)


def test_mode_same_is_the_default(monkeypatch):
    """No setting: the retry uses the same model and the old request; only two models are made."""
    out, roles, models = run_mode(monkeypatch, None, [AIMessage(content="", response_metadata=meta(1, 1))],
                                  [empty_with("notes ABC"), AIMessage(content="final", response_metadata=meta(1, 1))], [])
    assert out[-1].content == NO_DATA_HEADER + "final" and "empty_retry_mode" not in out[-1].response_metadata
    assert models["on"].seen[1][-1].content == "again"
    assert len(roles) == 2


def test_react_no_think_retry_uses_a_model_without_reasoning(monkeypatch):
    """The retry goes to a reasoning=False model of the same role, with the old notes in the request."""
    out, roles, models = run_mode(monkeypatch, "no_think", [AIMessage(content="", response_metadata=meta(1, 1))],
                                  [empty_with("notes ABC", done_reason="length")],
                                  [AIMessage(content="final", response_metadata=meta(1, 1))])
    assert [r.get("reasoning") for r in roles if r["name"] == "interpreter"] == [True, False]
    assert "notes ABC" in models["off"].seen[0][-1].content
    final = out[-1]
    assert final.content == NO_DATA_HEADER + "final" and final.response_metadata["empty_retry_mode"] == "no_think"
    record = model_record(out)
    assert record["answer_from_no_think_retry"] is True and record["length_stops"] == 1
    assert final.response_metadata["discarded_calls"][0]["reasoning"] is True


def test_react_no_think_empty_too_still_hands_off(monkeypatch):
    """If the no-think retry is empty as well, the work goes back to the explorer as before."""
    out, _, _ = run_mode(monkeypatch, "no_think", [AIMessage(content="", response_metadata=meta(1, 1))] * 2,
                         [empty_with("p"), AIMessage(content="final", response_metadata=meta(1, 1))],
                         [empty_with(None)])
    handoff = next(m for m in out if m.type == "human" and m.content.startswith(graph.HANDOFF_MARK))
    assert [c["reason"] for c in handoff.additional_kwargs["discarded_calls"]] == ["empty_retry", "handoff"]
    assert [c["reasoning"] for c in handoff.additional_kwargs["discarded_calls"]] == [True, False]
    assert out[-1].content == NO_DATA_HEADER + "final"


def test_missing_reasoning_says_no_notes():
    """A discarded reply with no reasoning_content gives "(no notes written)"."""
    text = graph.empty_retry_notes(empty_with(None), graph.EMPTY_RETRY_NOTES_DEFAULT)
    assert "(no notes written)" in text and "{notes}" not in text


def test_notes_are_at_most_plan_chars_long():
    """Only the tail of the reasoning is kept."""
    long = "x" * 5000 + "END"
    text = graph.empty_retry_notes(empty_with(long), "{notes}")
    assert len(text) == graph.PLAN_CHARS and text.endswith("END")


def test_length_stops_count_kept_and_discarded_calls():
    """done_reason "length" in a discarded call is counted once in the record."""
    kept = AIMessage(content="final", response_metadata={**meta(1, 1), "node": "interpret", "done_reason": "stop",
                                                         "discarded_calls": [graph.call_metadata(
                                                             empty_with("n", done_reason="length"), "interpret",
                                                             "empty_retry")]})
    assert model_record([HumanMessage(content="q"), kept])["length_stops"] == 1


def test_unknown_empty_retry_mode_is_an_error(monkeypatch):
    """A wrong value stops at graph construction with a clear message."""
    import pytest
    with pytest.raises(ValueError, match="empty_retry_mode"):
        run_mode(monkeypatch, "sometimes", [], [], [])


# ---------------------------------------------------------------- g5.1-f: no data read

def result(name, content, call_id="c1", status="success"):
    """A tool message."""
    return ToolMessage(content=content, name=name, tool_call_id=call_id, status=status)


def test_data_retrieved_needs_a_reading_tool_without_error():
    """summarize_results, describe_trends, compare_groups and query_results count; check_quality and errors do not."""
    ok = '{"rows": [[1]]}'
    error = '{"error": "unknown column", "error_type": "column"}'
    assert data_retrieved([result("query_results", ok)])
    assert data_retrieved([result("describe_trends", error), result("compare_groups", ok, "c2")])
    assert not data_retrieved([result("describe_trends", error), result("check_quality", ok, "c2")])
    # 3/11: for the header, check_quality counts (its missing-value counts are data); describe_database does not
    assert data_retrieved([result("describe_trends", error), result("check_quality", ok, "c2")], ANSWER_TOOLS)
    assert not data_retrieved([result("describe_database", ok, "c2")], ANSWER_TOOLS)
    assert not data_retrieved([result("summarize_results", "Error: ValidationError: x", status="error")])
    assert not data_retrieved([HumanMessage(content="q")])
    assert step_error_text(result("query_results", error)) == "unknown column"
    assert step_error_text(result("query_results", '{"warning": "w", "rows": []}')) is None


def run_with_tools(monkeypatch, toolbox, explorer_replies, interpreter_replies):
    """The ReAct graph with the real tools of the fake database and scripted models; returns the messages."""
    scripted = {"explorer": list(explorer_replies), "interpreter": list(interpreter_replies)}
    monkeypatch.setattr(graph, "make_chat_model", lambda provider, role, seed: FakeModel(scripted[role["name"]]))
    settings = {"provider": "x", "limits": {"empty_retries": 1, "handoffs": 1},
                "roles": {"explorer": {"name": "explorer"}, "interpreter": {"name": "interpreter"}}}
    tools = [t for t in make_langchain_tools(toolbox) if t.name in graph.GRAPH_TOOLS]
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=tools,
                              roles={"explore": "go {DATA_COLLECTED}", "interpret": "write"})
    return make_graph(analyst, seed=1).invoke({"messages": [HumanMessage(content="q")]})["messages"]


def calls(name, args, call_id="c1"):
    """An explorer reply that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}], response_metadata=meta(10, 1))


def done():
    """An explorer reply that ends the collection."""
    return AIMessage(content=DATA_COLLECTED, response_metadata=meta(10, 1))


def test_an_answer_after_only_failed_reading_calls_starts_with_the_header(monkeypatch, toolbox):
    """Every reading tool failed: the code puts the header before the interpreter's text."""
    messages = run_with_tools(
        monkeypatch, toolbox,
        [calls("describe_trends", {"value": "no_such_column", "x": "step"}), done()],
        [AIMessage(content="The level is 400 [tool].", response_metadata=meta(10, 1))])
    assert messages[-1].content == NO_DATA_HEADER + "The level is 400 [tool]."
    assert NO_DATA_HEADER.startswith("**Data not retrieved.**") and NO_DATA_HEADER.endswith("\n\n")


def test_an_answer_after_a_good_reading_call_has_no_header(monkeypatch, toolbox):
    """One query worked: the interpreter's text is kept as it is."""
    messages = run_with_tools(
        monkeypatch, toolbox,
        [calls("query_results", {"sql": "SELECT COUNT(*) AS n FROM runs"}), done()],
        [AIMessage(content="There are 8 runs [tool].", response_metadata=meta(10, 1))])
    assert messages[-1].content == "There are 8 runs [tool]."


def test_check_quality_alone_is_not_data_and_a_tool_call_reply_gets_no_header(monkeypatch, toolbox):
    """check_quality does not read the data; an interpreter reply that asks for a tool is not an answer."""
    messages = run_with_tools(
        monkeypatch, toolbox,
        [calls("check_quality", {"table": "runs"}), done()],
        [calls("query_results", {"sql": "SELECT 1 AS n"}, "c2"),
         AIMessage(content="One [tool].", response_metadata=meta(10, 1))])
    asking = [m for m in messages if m.type == "ai" and m.tool_calls and m.response_metadata.get("node") == "interpret"]
    assert asking and asking[0].content == ""
    assert messages[-1].content == "One [tool]."          # the query worked in the meantime
