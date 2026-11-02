"""Tests of the ReWOO graph (plan -> execute in code -> analyze), with fake models and the fake database."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from bench import blind
from bench.run_bench import final_answer, model_record, repair_record, rewoo_record, with_reasoning
from sim_analyst import graph as graph_module
from sim_analyst import graphs, rewoo
from sim_analyst.checker import check_numbers
from sim_analyst.config import PROMPTS_DIR, SETTINGS_FILE, load_yaml
from sim_analyst.rewoo import Plan, Step, make_rewoo_graph, run_steps, step_error
from sim_analyst.tool_schemas import make_langchain_tools

AMOUNT = "config.strategies[0].amount"
GOOD_SUMMARY = {"tool": "summarize_results", "why": "means at the last step",
                "args": {"value": "value", "group_by": AMOUNT, "time": "step", "at": [10],
                         "filters": {"quantity": "x", "experiment": "exp_a"}}}
BAD_COLUMN = {"tool": "summarize_results", "why": "wrong column",
              "args": {"value": "no_such_column"}}


# ---------------------------------------------------------------- the plan schema

def test_step_rejects_an_unknown_tool():
    """Only the four data tools can be planned: calculate and anything else is refused by the schema."""
    with pytest.raises(ValidationError):
        Step.model_validate({"tool": "drop_table", "args": {}, "why": "x"})
    with pytest.raises(ValidationError):
        Step.model_validate({"tool": "calculate", "args": {"expression": "1+1"}, "why": "x"})


def test_plan_has_between_1_and_8_steps():
    """0 and 9 steps are refused, 1 and 8 are fine."""
    step = {"tool": "describe_trends", "args": {"value": "value", "time": "step"}, "why": "x"}
    for n in (0, 9):
        with pytest.raises(ValidationError):
            Plan.model_validate({"steps": [step] * n})
    for n in (1, 8):
        assert len(Plan.model_validate({"steps": [step] * n}).steps) == n


def test_step_args_are_checked_with_the_schema_of_the_tool():
    """Missing required argument, unknown argument and wrong type are all reported; a good step has no error."""
    assert step_error(Step.model_validate(GOOD_SUMMARY)) is None
    missing = step_error(Step(tool="summarize_results", args={}, why="x"))
    assert missing and "value" in missing
    unknown = step_error(Step(tool="summarize_results", args={"value": "value", "colum": "x"}, why="x"))
    assert unknown and "colum" in unknown
    missing_b = step_error(Step(tool="compare_groups", args={"value": "value", "group_by": "g", "a": 1}, why="x"))
    assert missing_b and "b" in missing_b
    wrong_type = step_error(Step(tool="describe_trends", args={"value": "v", "time": "t", "step": "abc"}, why="x"))
    assert wrong_type and "step" in wrong_type


# ---------------------------------------------------------------- fakes

def meta(read=100, written=10, read_s=1.0, write_s=2.0):
    """Ollama-like metadata of one call."""
    return {"prompt_eval_count": read, "eval_count": written,
            "prompt_eval_duration": int(read_s * 1e9), "eval_duration": int(write_s * 1e9)}


def plan_reply(steps, **kw):
    """A planner reply as with_structured_output(..., include_raw=True) gives it, for a valid schema."""
    text = json.dumps({"steps": steps})
    raw = AIMessage(content=text, additional_kwargs={"reasoning_content": "planner thinks"},
                    response_metadata=meta(**kw))
    return {"raw": raw, "parsed": Plan.model_validate({"steps": steps}), "parsing_error": None}


def unparsed_reply(steps, **kw):
    """A planner reply the schema refused: no parsed plan, but the raw JSON can still be read."""
    text = json.dumps({"steps": steps})
    try:
        Plan.model_validate({"steps": steps})
    except ValidationError as exc:
        error = exc
    raw = AIMessage(content=text, additional_kwargs={"reasoning_content": "planner thinks"},
                    response_metadata=meta(**kw))
    return {"raw": raw, "parsed": None, "parsing_error": error}


class FakeStructured:
    """What with_structured_output returns: scripted replies, and a record of the messages it saw."""

    def __init__(self, replies):
        """Keep the replies to give, in order."""
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        """Return the next scripted reply."""
        self.seen.append(messages)
        return self.replies.pop(0)


class FakeChat:
    """A chat model with scripted replies; remembers which tools were bound at each call."""

    def __init__(self, replies=(), structured=None):
        """Scripted plain replies and/or a structured-output runnable."""
        self.replies, self.structured, self.tools_at_call, self.seen = list(replies), structured, [], []
        self.structured_args = None

    def with_structured_output(self, schema, method=None, include_raw=False):
        """Remember how it was asked for; return the scripted runnable."""
        self.structured_args = (schema, method, include_raw)
        return self.structured

    def bind_tools(self, tools):
        """A view of the same model that remembers the names of the bound tools."""
        parent, names = self, [t.name for t in tools]
        return SimpleNamespace(invoke=lambda messages: parent.invoke(messages, names))

    def invoke(self, messages, tools=()):
        """Return the next scripted reply."""
        self.tools_at_call.append(list(tools))
        self.seen.append(messages)
        return self.replies.pop(0)


def calc_call(expression, call_id):
    """An analyst reply that asks for the calculate tool."""
    return AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": expression},
                                              "id": call_id}], response_metadata=meta(200, 20, 2.0, 3.0))


def final_reply(text="final 100", **kw):
    """An analyst reply with the final answer."""
    return AIMessage(content=text, additional_kwargs={"reasoning_content": "analyst thinks"},
                     response_metadata=meta(**{"read": 300, "written": 40, **kw}))


def build(toolbox, monkeypatch, planner_replies, analyst_replies, calculate_calls=8):
    """The ReWOO graph on the fake database with scripted models; returns (graph, planner, analyst)."""
    planner = FakeChat(structured=FakeStructured(planner_replies))
    analyst_model = FakeChat(replies=analyst_replies)
    chats = {"planner": planner, "analyst": analyst_model}
    monkeypatch.setattr(rewoo, "make_chat_model", lambda provider, role, seed: chats[role["name"]])
    settings = {"provider": "x", "limits": {"empty_retries": 1, "calculate_calls": calculate_calls},
                "roles": {"planner": {"name": "planner"}, "analyst": {"name": "analyst"}}}
    tools = [t for t in make_langchain_tools(toolbox) if t.name in graph_module.GRAPH_TOOLS]
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=tools,
                              roles={"plan": "plan it", "analyze": "write it", "empty_retry": "again"})
    return make_rewoo_graph(analyst, seed=1), planner, analyst_model


def ask(graph):
    """Run the graph on one question; return its messages."""
    return graph.invoke({"messages": [HumanMessage(content="q")]})["messages"]


# ---------------------------------------------------------------- planner

def test_planner_is_asked_for_a_plan_with_the_json_schema(toolbox, monkeypatch):
    """The planner is the structured-output runnable of Plan (json_schema, raw message included)."""
    graph, planner, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply()])
    ask(graph)
    assert planner.structured_args == (Plan, "json_schema", True)


def test_first_invalid_plan_is_asked_again_with_the_error(toolbox, monkeypatch):
    """A refused plan gets ONE more request, with the error text; the valid second plan is executed."""
    first = unparsed_reply([], read=100, written=11)
    second = plan_reply([GOOD_SUMMARY], read=120, written=12)
    graph, planner, _ = build(toolbox, monkeypatch, [first, second], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 2
    retry_request = planner.structured.seen[1][-1]
    assert retry_request.type == "human" and "steps" in retry_request.content   # the schema error is named
    plan = messages[1]
    assert [c["name"] for c in plan.tool_calls] == ["summarize_results"]
    assert len(plan.response_metadata["plan_rejections"]) == 1
    assert plan.response_metadata["plan_retries"] == 1
    assert [d["reason"] for d in plan.response_metadata["discarded_calls"]] == ["plan_retry"]
    record = rewoo_record(messages)
    assert record["plan_rejections"] == 1 and record["plan_retries"] == 1 and record["steps_planned"] == 1


def test_two_invalid_plans_keep_only_the_valid_steps(toolbox, monkeypatch):
    """After the second failure the valid steps are executed and both rejections are recorded."""
    first = plan_reply([GOOD_SUMMARY, {"tool": "summarize_results", "args": {}, "why": "no value"}])
    second = unparsed_reply([GOOD_SUMMARY, {"tool": "drop_table", "args": {}, "why": "unknown tool"}])
    graph, planner, _ = build(toolbox, monkeypatch, [first, second], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 2           # never a third request
    plan = messages[1]
    assert [c["name"] for c in plan.tool_calls] == ["summarize_results"]
    assert len(plan.response_metadata["plan_rejections"]) == 2
    assert rewoo_record(messages)["plan_rejections"] == 2


def test_a_valid_first_plan_has_no_rejection(toolbox, monkeypatch):
    """Nothing refused: no retry, no discarded call."""
    graph, planner, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 1
    assert rewoo_record(messages)["plan_rejections"] == 0 and rewoo_record(messages)["plan_retries"] == 0


def with_why(step, **extra):
    """A copy of a step with 'why' (and other keys) added to its args."""
    return {**step, "args": {**step["args"], **extra}}


def test_why_in_args_is_removed_not_refused(toolbox, monkeypatch):
    """The planner repeats the reason inside args: the key is dropped, the step runs, the plan is accepted."""
    graph, planner, _ = build(toolbox, monkeypatch,
                              [plan_reply([with_why(GOOD_SUMMARY, why="means at the last step")])], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 1           # no retry
    plan = messages[1]
    assert plan.tool_calls[0]["args"] == GOOD_SUMMARY["args"]
    assert plan.response_metadata["plan_rejections"] == []
    assert plan.response_metadata["why_in_args"] == 1
    assert rewoo_record(messages)["why_in_args"] == 1


def test_why_in_args_fills_an_empty_why(toolbox, monkeypatch):
    """If the step's own why is empty, the one found in args is used as its reason."""
    step = {**with_why(GOOD_SUMMARY, why="reason from args"), "why": ""}
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply([step])], [final_reply()])
    plan = ask(graph)[1]
    assert "reason from args" in plan.content


def test_why_in_args_does_not_replace_a_given_why(toolbox, monkeypatch):
    """A non-empty step why wins over the one in args."""
    step = with_why(GOOD_SUMMARY, why="other text")
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply([step])], [final_reply()])
    plan = ask(graph)[1]
    assert "means at the last step" in plan.content and "other text" not in plan.content


def test_why_is_counted_per_step(toolbox, monkeypatch):
    """Two steps with a why in args count 2; a step without it does not count."""
    steps = [with_why(GOOD_SUMMARY, why="a"), GOOD_SUMMARY, with_why(GOOD_SUMMARY, why="b")]
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply(steps)], [final_reply()])
    assert rewoo_record(ask(graph))["why_in_args"] == 2


def test_other_unknown_arguments_are_still_refused(toolbox, monkeypatch):
    """Only 'why' is forgiven: another unknown key refuses the step, even next to a why."""
    bad = with_why(GOOD_SUMMARY, why="x", colour="red")
    graph, planner, _ = build(toolbox, monkeypatch, [plan_reply([bad]), plan_reply([GOOD_SUMMARY])], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 2
    assert "colour" in messages[1].response_metadata["plan_rejections"][0][0]


def test_no_why_in_args_counts_zero(toolbox, monkeypatch):
    """A clean plan reports why_in_args 0."""
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply()])
    assert rewoo_record(ask(graph))["why_in_args"] == 0


# ---------------------------------------------------------------- execute

def plan_message(*steps):
    """A plan message as the plan node writes it."""
    return AIMessage(content="", tool_calls=[{"name": s["tool"], "args": s["args"], "id": f"step_{i}",
                                              "type": "tool_call"} for i, s in enumerate(steps, 1)])


def test_run_steps_needs_no_model_and_keeps_order(toolbox):
    """Plain Python: one ToolMessage per step, in order, with the id of its call."""
    tools = {t.name: t for t in make_langchain_tools(toolbox)}
    plan = plan_message(GOOD_SUMMARY, {"tool": "query_results", "args": {"sql": "SELECT COUNT(*) AS n FROM runs"}})
    out = run_steps(tools, plan)
    assert [m.tool_call_id for m in out] == ["step_1", "step_2"]
    assert [m.name for m in out] == ["summarize_results", "query_results"]
    assert json.loads(out[0].content)["rows"][0][:7] == [10, 10, 3, 100, 1, 99, 101]
    assert json.loads(out[1].content)["rows"] == [[8]]


def test_a_failing_step_leaves_an_error_and_the_next_steps_run(toolbox):
    """An error result stays in the conversation; there is no correction; the later step still runs."""
    tools = {t.name: t for t in make_langchain_tools(toolbox)}
    out = run_steps(tools, plan_message(BAD_COLUMN, GOOD_SUMMARY))
    assert json.loads(out[0].content)["error_type"] == "column"
    assert "rows" in json.loads(out[1].content)


def test_a_tool_that_raises_becomes_an_error_message():
    """An unexpected exception does not stop the run: it becomes a tool message that starts with 'Error'."""
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("broken")
    tools = {"query_results": StructuredTool.from_function(boom, name="query_results", description="d")}
    ok_plan = plan_message({"tool": "query_results", "args": {}})
    out = run_steps(tools, ok_plan)
    assert out[0].content.startswith("Error") and "broken" in out[0].content and out[0].tool_call_id == "step_1"


# ---------------------------------------------------------------- transcript

def test_transcript_has_the_shape_of_the_v2_graph(toolbox, monkeypatch):
    """human, plan (tool calls + reasoning + node), one tool message per call, final answer."""
    plan = plan_reply([GOOD_SUMMARY, BAD_COLUMN], read=100, written=10)
    graph, _, _ = build(toolbox, monkeypatch, [plan], [final_reply("The mean is 100 [tool].")])
    messages = ask(graph)
    assert [m.type for m in messages] == ["human", "ai", "tool", "tool", "ai"]
    plan_msg = messages[1]
    assert plan_msg.response_metadata["node"] == "plan" and plan_msg.response_metadata["eval_count"] == 10
    assert plan_msg.additional_kwargs["reasoning_content"] == "planner thinks"
    assert [c["id"] for c in plan_msg.tool_calls] == [m.tool_call_id for m in messages[2:4]]
    assert "means at the last step" in plan_msg.content       # the reason of each step stays readable
    assert messages[-1].response_metadata["node"] == "analyze"


def test_blind_and_number_checker_work_on_a_rewoo_transcript(toolbox, monkeypatch):
    """The final answer is found in the transcript text and its numbers are traced to the tool results."""
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply("The mean is 100 [tool].")])
    messages = ask(graph)
    transcript = "\n".join(with_reasoning(m) for m in messages)
    assert blind.final_answer_of(transcript) == "The mean is 100 [tool]."
    answer = final_answer(messages)
    sources = [m.content for m in messages if m.type == "tool"]
    assert check_numbers(answer, sources)["numbers_untraceable"] == 0
    assert check_numbers("The mean is 4242.", sources)["numbers_untraceable"] == 1


def test_measures_per_node_and_by_error_kind(toolbox, monkeypatch):
    """per_node has plan and analyze; steps_failed is by error kind; calculate calls are counted."""
    planner_reply = plan_reply([GOOD_SUMMARY, BAD_COLUMN], read=100, written=10, read_s=1.0, write_s=2.0)
    graph, _, _ = build(toolbox, monkeypatch, [planner_reply],
                        [calc_call("1+1", "c1"), final_reply()])
    messages = ask(graph)
    record = rewoo_record(messages)
    assert record["steps_planned"] == 2 and record["steps_failed"] == 1
    assert record["steps_failed_per_kind"] == {"column": 1}
    assert record["calculate_calls"] == 1
    nodes = model_record(messages)["per_node"]
    assert nodes["plan"] == {"model_calls": 1, "tokens_read": 100, "tokens_written": 10,
                             "seconds_reading": 1.0, "seconds_writing": 2.0}
    assert nodes["analyze"]["model_calls"] == 2 and nodes["analyze"]["tokens_read"] == 500


# ---------------------------------------------------------------- analyst

def test_the_analyst_has_only_calculate(toolbox, monkeypatch):
    """Every analyst call is made with calculate bound, and nothing else."""
    graph, _, analyst_model = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])],
                                    [calc_call("2*3", "c1"), final_reply()])
    messages = ask(graph)
    assert analyst_model.tools_at_call == [["calculate"], ["calculate"]]
    assert json.loads(messages[-2].content)["result"] == 6


def test_the_calculate_loop_stops_at_the_limit(toolbox, monkeypatch):
    """After limits.calculate_calls calculations the analyst is called without tools and must answer."""
    replies = [calc_call("1+1", "c1"), calc_call("2+2", "c2"), final_reply("done 4")]
    graph, _, analyst_model = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], replies, calculate_calls=2)
    messages = ask(graph)
    assert sum(1 for m in messages if m.type == "tool" and m.name == "calculate") == 2
    assert analyst_model.tools_at_call == [["calculate"], ["calculate"], []]
    assert messages[-1].content == "done 4"


def test_an_empty_analyst_reply_is_asked_again_once(toolbox, monkeypatch):
    """Empty reply -> one more request; the discarded call is kept and counted; no handoff exists."""
    replies = [AIMessage(content="", response_metadata=meta(400, 300, 4.0, 50.0)), final_reply("answer 5")]
    graph, _, analyst_model = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], replies)
    messages = ask(graph)
    assert messages[-1].content == "answer 5"
    assert analyst_model.seen[1][-1].content == "again"      # the empty_retry request of the roles
    record = model_record(messages)
    assert record["empty_retries"] == 1 and record["handoffs"] == 0
    assert record["per_node"]["analyze"]["model_calls"] == 2


def test_two_empty_analyst_replies_end_without_answer_and_without_handoff(toolbox, monkeypatch):
    """After the retry there is no handoff: the run ends with an empty answer."""
    replies = [AIMessage(content="", response_metadata=meta()) for _ in range(2)]
    graph, _, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], replies)
    messages = ask(graph)
    assert messages[-1].content == "" and not any(m.type == "human" and m.content.startswith("[HANDOFF]")
                                                  for m in messages)
    assert model_record(messages)["empty_answer"] is True


# ---------------------------------------------------------------- settings and graph choice

def test_the_graph_setting_picks_the_graph(monkeypatch):
    """settings.graph chooses the builder; an unknown name is an error."""
    assert set(graphs.GRAPHS) == {"v2", "rewoo", "rewoo_repair"} and graphs.ANSWER_ROLE["rewoo_repair"] == "analyst"
    monkeypatch.setitem(graphs.GRAPHS, "v2", lambda analyst, seed: "v2 graph")
    monkeypatch.setitem(graphs.GRAPHS, "rewoo", lambda analyst, seed: "rewoo graph")
    analyst = lambda name: SimpleNamespace(domain=SimpleNamespace(settings={"graph": name}))
    assert graphs.build_graph(analyst("v2"), 1) == "v2 graph"
    assert graphs.build_graph(analyst("rewoo"), 1) == "rewoo graph"
    with pytest.raises(ValueError):
        graphs.build_graph(analyst("nope"), 1)


def test_run_settings_declare_the_graph_and_the_new_roles():
    """The default graph stays v2; planner and analyst roles exist; calculate_calls is 8."""
    settings = load_yaml(SETTINGS_FILE)
    assert settings["graph"] == "v2"
    assert settings["limits"]["calculate_calls"] == 8
    for name in ("planner", "analyst"):
        role = settings["roles"][name]
        assert role["reasoning"] is True and role["temperature"] == 1.0
        assert (role["top_k"], role["top_p"], role["max_tokens"]) == (64, 0.95, 8192)
        assert role["model"] == settings["roles"]["interpreter"]["model"]


def test_method_v4_differs_from_v3_only_on_line_45():
    """The only change of the method is the rule on where numbers come from."""
    v3 = (PROMPTS_DIR / "method_v3.md").read_text(encoding="utf-8").splitlines()
    v4 = (PROMPTS_DIR / "method_v4.md").read_text(encoding="utf-8").splitlines()
    assert len(v3) == len(v4)
    changed = [i + 1 for i, (a, b) in enumerate(zip(v3, v4)) if a != b]
    assert changed == [45]
    assert v4[44] == "- Every number in the answer must come from a tool result, the context or the calculate tool."


def test_roles_v4_has_plan_analyze_and_empty_retry():
    """The three requests of the ReWOO roles; the plan request says the results are not visible yet."""
    roles = load_yaml(PROMPTS_DIR / "roles_v4.yaml")
    assert set(roles) == {"plan", "analyze", "empty_retry", "repair", "repair_retry"}
    assert "will not see the results" in roles["plan"]


# ---------------------------------------------------------------- what the planner is told about the tools

def test_planner_sees_the_data_tools_and_not_calculate(toolbox, monkeypatch):
    """The plan request lists each data tool with its description and every argument name; calculate is absent."""
    from sim_analyst.tool_schemas import ARGS_SCHEMAS, DESCRIPTIONS
    graph, planner, _ = build(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply()])
    ask(graph)
    text = "\n".join(str(m.content) for m in planner.structured.seen[0])
    assert "Data tools you can plan" in text
    for name in ("query_results", "summarize_results", "describe_trends", "compare_groups", "check_quality"):
        assert name in text and DESCRIPTIONS[name] in text
        for argument in ARGS_SCHEMAS[name].model_fields:
            assert f'"{argument}"' in text, (name, argument)
    assert "calculate" not in text                                   # not offered, not even mentioned
    assert "### describe_database" not in text                       # the profile is already in the system message
    assert text.index("plan it") < text.index("Data tools you can plan")   # appended to the plan request


def test_the_tools_section_is_also_in_the_retry(toolbox, monkeypatch):
    """The second request keeps the plan request, so the tools are still listed."""
    first = unparsed_reply([])
    graph, planner, _ = build(toolbox, monkeypatch, [first, plan_reply([GOOD_SUMMARY])], [final_reply()])
    ask(graph)
    assert all("Data tools you can plan" in "\n".join(str(m.content) for m in seen)
               for seen in planner.structured.seen)


def test_check_quality_is_a_valid_plan_step_and_runs(toolbox):
    """check_quality is one of the planner's tools: its arguments are checked and the step runs in code."""
    step = Step(tool="check_quality", args={"table": "series", "group_by": "quantity"}, why="NULLs and duplicates")
    assert step_error(step) is None and "check_quality" in rewoo.DATA_TOOLS
    assert "check_quality: unknown argument" in step_error(Step(tool="check_quality", args={"tabel": "x"}, why="w"))
    assert step_error(Step(tool="check_quality", args={}, why="w")) is not None          # table is required
    tools = {t.name: t for t in make_langchain_tools(toolbox) if t.name in graph_module.GRAPH_TOOLS}
    out = run_steps(tools, plan_message({"tool": "check_quality", "args": {"table": "runs"}}))
    assert json.loads(out[0].content)["rows"] == 8


def test_unit_and_period_are_valid_plan_arguments():
    """The new arguments pass the plan check."""
    assert step_error(Step(tool="summarize_results", args={"value": "v", "unit": ["a", "b"]}, why="w")) is None
    assert step_error(Step(tool="compare_groups", args={"value": "v", "group_by": "g", "a": 1, "b": 2,
                                                         "unit": "a"}, why="w")) is None
    assert step_error(Step(tool="describe_trends", args={"value": "v", "time": "d", "period": "week"}, why="w")) is None
    assert step_error(Step(tool="describe_trends", args={"value": "v", "time": "d", "period": "year"}, why="w")) is not None


# ---------------------------------------------------------------- rewoo_repair

REPAIR_ROLE = "Some steps of your plan failed. Write only the steps that replace them."
REPAIR_RETRY_ROLE = "Your repair steps were refused for these reasons:\n{problems}\nWrite the replacement steps again."
GOOD_QUERY = {"tool": "query_results", "why": "count the runs", "args": {"sql": "SELECT COUNT(*) AS n FROM runs"}}
EXP_A = {"quantity": "x", "experiment": "exp_a"}
BAD_COLUMN_2 = {"tool": "summarize_results", "why": "wrong again", "args": {"value": "still_no_such_column"}}


def build_repair(toolbox, monkeypatch, planner_replies, analyst_replies, rounds=1, graph_name="rewoo_repair"):
    """The graph on the fake database; the planner is used by plan AND repair. Returns (graph, planner, analyst)."""
    planner = FakeChat(structured=FakeStructured(planner_replies))
    analyst_model = FakeChat(replies=analyst_replies)
    chats = {"planner": planner, "analyst": analyst_model}
    monkeypatch.setattr(rewoo, "make_chat_model", lambda provider, role, seed: chats[role["name"]])
    settings = {"provider": "x", "limits": {"empty_retries": 1, "calculate_calls": 8, "repair_rounds": rounds},
                "roles": {"planner": {"name": "planner"}, "analyst": {"name": "analyst"}}}
    tools = [t for t in make_langchain_tools(toolbox) if t.name in graph_module.GRAPH_TOOLS]
    analyst = SimpleNamespace(domain=SimpleNamespace(settings=settings), system_text="S", tools=tools,
                              roles={"plan": "plan it", "analyze": "write it", "empty_retry": "again",
                                     "repair": REPAIR_ROLE,
                                     "repair_retry": REPAIR_RETRY_ROLE})
    return graphs.GRAPHS[graph_name](analyst, 1), planner, analyst_model


def nodes_of(messages):
    return [m.response_metadata.get("node") for m in messages if m.type == "ai"]


def test_no_failed_step_means_no_repair_and_the_same_messages_as_rewoo(toolbox, monkeypatch):
    """Notes are not failures: the repair node is never visited and the transcript equals the rewoo one."""
    with_notes = {"tool": "summarize_results", "why": "pooled", "args": {"value": "value", "group_by": AMOUNT,
                                                                         "filters": EXP_A}}
    runs = []
    for name in ("rewoo", "rewoo_repair"):
        graph, planner, _ = build_repair(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY, with_notes])],
                                         [final_reply()], graph_name=name)
        runs.append((ask(graph), planner))
    (plain, _), (repaired, planner) = runs
    assert len(planner.structured.seen) == 1 and "repair" not in nodes_of(repaired)
    strip = lambda ms: [(m.type, str(m.content), getattr(m, "tool_call_id", None)) for m in ms]
    assert strip(plain) == strip(repaired)


def test_rewoo_with_a_failed_step_is_unchanged_by_the_new_graph(toolbox, monkeypatch):
    """The plain graph never repairs: planner called once, the error stays in the conversation."""
    graph, planner, _ = build_repair(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY, BAD_COLUMN])],
                                     [final_reply()], graph_name="rewoo")
    messages = ask(graph)
    assert len(planner.structured.seen) == 1 and nodes_of(messages) == ["plan", "analyze"]


def test_one_failed_step_is_repaired_once_with_the_error_in_the_prompt(toolbox, monkeypatch):
    """The repair prompt has the error text, the ok step as a short line, the system message and the tools."""
    graph, planner, analyst = build_repair(
        toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY, BAD_COLUMN]), plan_reply([GOOD_QUERY], read=50, written=5)],
        [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 2
    seen = planner.structured.seen[1]
    assert seen[0].content == "S" and seen[1].content == "q"                      # system message, then the question
    text = str(seen[-1].content)
    assert REPAIR_ROLE in text and "unknown column 'no_such_column'" in text       # the full error
    assert "summarize_results" in text and "ok" in text and "means at the last step" not in text
    assert '"sql"' not in text.split("Data tools you can plan")[0]                  # no result of ok steps is repeated
    assert "Data tools you can plan" in text and "check_quality" in text
    assert nodes_of(messages) == ["plan", "repair", "analyze"]
    assert [m.type for m in messages] == ["human", "ai", "tool", "tool", "ai", "tool", "ai"]
    ids = [m.tool_call_id for m in messages if m.type == "tool"]
    assert len(ids) == len(set(ids)) == 3 and ids[2] != "step_1"
    assert [m.name for m in messages if m.type == "tool"] == ["summarize_results", "summarize_results", "query_results"]


def test_only_the_new_steps_run_and_the_analyst_sees_everything(toolbox, monkeypatch):
    """The ok step is not run again; the analyst's input holds first-plan results, the error and the repair result."""
    graph, _, analyst = build_repair(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY, BAD_COLUMN]),
                                                            plan_reply([GOOD_QUERY])], [final_reply()])
    messages = ask(graph)
    assert sum(1 for m in messages if m.type == "tool" and m.name == "summarize_results") == 2
    seen = "\n".join(str(m.content) for m in analyst.seen[0])
    assert '"rows": [[10, 10, 3, 100' in seen and "unknown column" in seen and '"rows": [[8]]' in seen
    assert seen.index("unknown column") < seen.index('"rows": [[8]]')              # plan, results, repair, results


def test_a_repair_that_fails_again_gets_no_second_round(toolbox, monkeypatch):
    """With repair_rounds = 1 the second failure goes to the analyst; the measures count it."""
    graph, planner, _ = build_repair(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY, BAD_COLUMN]),
                                                            plan_reply([BAD_COLUMN_2])], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 2 and nodes_of(messages) == ["plan", "repair", "analyze"]
    record = repair_record(messages)
    assert record == {"repair_rounds_done": 1, "steps_repaired": 1, "steps_failed_before_repair": 1,
                      "steps_failed_after_repair": 1, "repair_rejections": 0}


def test_two_rounds_are_possible_when_the_setting_allows(toolbox, monkeypatch):
    """repair_rounds = 2: a second repair happens, and its prompt shows both earlier rounds."""
    graph, planner, _ = build_repair(toolbox, monkeypatch, [plan_reply([BAD_COLUMN]), plan_reply([BAD_COLUMN_2]),
                                                            plan_reply([GOOD_QUERY])], [final_reply()], rounds=2)
    messages = ask(graph)
    assert nodes_of(messages) == ["plan", "repair", "repair", "analyze"]
    third = str(planner.structured.seen[2][-1].content)
    assert "no_such_column" in third and "still_no_such_column" in third


def test_an_invalid_repair_plan_is_asked_again_once_and_the_rejection_is_recorded(toolbox, monkeypatch):
    """Unknown argument in the repair plan: one retry with the problem; the valid steps run; counted as repair_rejections."""
    bad = {"tool": "query_results", "why": "x", "args": {"sqll": "SELECT 1"}}
    graph, planner, _ = build_repair(toolbox, monkeypatch, [plan_reply([BAD_COLUMN]), plan_reply([bad]),
                                                            plan_reply([GOOD_QUERY])], [final_reply()])
    messages = ask(graph)
    assert len(planner.structured.seen) == 3
    retry = str(planner.structured.seen[2][-1].content)
    assert "unknown argument" in retry and "Your repair steps were refused" in retry
    assert "Write the whole plan again" not in retry
    repair = next(m for m in messages if m.type == "ai" and m.response_metadata.get("node") == "repair")
    assert len(repair.response_metadata["repair_rejections"]) == 1 and repair.response_metadata["repair_retries"] == 1
    assert "plan_rejections" not in repair.response_metadata
    assert [c["name"] for c in repair.tool_calls] == ["query_results"]
    record = repair_record(messages)
    assert record["repair_rejections"] == 1 and record["steps_repaired"] == 1 and record["steps_failed_after_repair"] == 0
    assert model_record(messages)["per_node"]["repair"]["model_calls"] == 2        # the discarded call is counted


def test_repair_measures_without_a_repair(toolbox, monkeypatch):
    """No failure: zero everywhere."""
    graph, _, _ = build_repair(toolbox, monkeypatch, [plan_reply([GOOD_SUMMARY])], [final_reply()])
    assert repair_record(ask(graph)) == {"repair_rounds_done": 0, "steps_repaired": 0, "steps_failed_before_repair": 0,
                                         "steps_failed_after_repair": 0, "repair_rejections": 0}


def test_repair_time_and_tokens_are_per_node(toolbox, monkeypatch):
    """The repair node has its own entry in per_node."""
    graph, _, _ = build_repair(toolbox, monkeypatch, [plan_reply([BAD_COLUMN]),
                                                      plan_reply([GOOD_QUERY], read=70, written=7)], [final_reply()])
    nodes = model_record(ask(graph))["per_node"]
    assert nodes["repair"]["tokens_read"] == 70 and nodes["repair"]["tokens_written"] == 7


def test_unknown_graph_name_is_a_clear_error():
    analyst = SimpleNamespace(domain=SimpleNamespace(settings={"graph": "rewoo_replan"}))
    with pytest.raises(ValueError, match="unknown graph 'rewoo_replan'.*rewoo_repair"):
        graphs.build_graph(analyst, 1)


def test_repair_retry_role_text_is_the_agreed_one_and_the_default_does_not_say_whole_plan():
    roles = load_yaml(PROMPTS_DIR / "roles_v4.yaml")
    assert roles["repair_retry"].split() == ("Your repair steps were refused for these reasons: {problems}. "
                                              "Write the replacement steps again, fixing them; do not repeat "
                                              "the steps that worked.").split()
    assert "whole plan" not in rewoo.REPAIR_RETRY_DEFAULT and "{problems}" in rewoo.REPAIR_RETRY_DEFAULT


def test_the_repair_role_text_has_no_domain_words():
    """Generic wording only."""
    roles = load_yaml(PROMPTS_DIR / "roles_v4.yaml")
    text = (roles["repair"] + " " + roles["repair_retry"]).lower()
    assert "failed" in text and "error" in text
    for word in ("run", "customer", "station", "rental", "shell", "toy", "simulation", "experiment", "amount"):
        assert word not in text.split() and f" {word} " not in f" {text} ", word
    assert "{" not in roles["repair"]
