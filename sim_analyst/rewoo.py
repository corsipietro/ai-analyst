"""The ReWOO graph: a planner writes ALL the tool calls, code runs them, an analyst writes the answer.

Compared with graph.py (explorer <-> tools, then interpreter), the model never sees a result
while it is deciding what to ask for ("reasoning WithOut Observation"):
1. plan: the planner (reasoning on) returns a Plan, a list of 1 to 8 Steps (tool, args, why) written
   as JSON that follows the Pydantic schema. Each step's args are checked with the argument schema
   of its tool. A refused plan is asked ONCE more, with the error text; after a second refusal the
   valid steps are kept. The rejections are recorded, so the bench can count them.
2. execute: plain Python, no model. The steps run in order through the same LangChain tools as
   graph.py. An error stays in the conversation as a result: nothing corrects it in this graph.
3. analyze: the analyst (reasoning on) writes the answer. It has ONLY the calculate tool, at most
   limits.calculate_calls times; after that it is called with no tool, so it must answer. An empty
   reply is asked again (limits.empty_retries), with no handoff.

The plan is saved as ONE model message whose tool_calls are the steps, followed by one tool message
per step: transcripts, number checker and blind grading work as with graph.py.
"""
from __future__ import annotations

import json
from typing import Any, Literal, get_args

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from .graph import EMPTY_RETRY_DEFAULT, call_metadata, invoke_with_retries
from .llm import make_chat_model
from .tool_schemas import ARGS_SCHEMAS, DESCRIPTIONS

MAX_STEPS = 8
CALCULATE = "calculate"
PLAN_RETRY_DEFAULT = "Your plan was refused for these reasons:\n{problems}\nWrite the whole plan again, fixing them."
REPAIR_RETRY_DEFAULT = ("Your repair steps were refused for these reasons: {problems}. Write the replacement steps "
                        "again, fixing them; do not repeat the steps that worked.")
REPAIR_DEFAULT = ("Some steps of your plan failed. Read each error: it says what to change. Write only the steps that "
                  "replace the failed ones, with corrected arguments; do not repeat the steps that worked.")


class Step(BaseModel):
    """One tool call of the plan: which data tool, with which arguments, and why."""

    tool: Literal["query_results", "summarize_results", "describe_trends", "compare_groups", "check_quality"]
    args: dict[str, Any]
    why: str


class Plan(BaseModel):
    """What the planner returns: between 1 and MAX_STEPS steps, run in this order."""

    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)


DATA_TOOLS = get_args(Step.model_fields["tool"].annotation)   # the tools a plan may use


class RewooState(MessagesState):
    """Only the messages: the counters (calculate calls) are read from them."""


# ---------------------------------------------------------------- telling the planner about the tools

def without_titles(schema):
    """A JSON schema without the 'title' keys, which only repeat the names and cost tokens."""
    if isinstance(schema, dict):
        return {k: without_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [without_titles(v) for v in schema]
    return schema


def tools_section() -> str:
    """The section "Data tools you can plan": name, description and compact argument schema of each tool.

    with_structured_output only sends the schema of the Plan (tool is a name, args is any object),
    so the planner would not know what the tools do or which arguments they take. The text comes
    from tool_schemas.py, the same one the tools show to a model that calls them directly.
    """
    parts = ["## Data tools you can plan"]
    for name in DATA_TOOLS:
        schema = json.dumps(without_titles(ARGS_SCHEMAS[name].model_json_schema()), separators=(",", ":"))
        parts.append(f"### {name}\n{DESCRIPTIONS[name]}\nArguments (JSON schema): {schema}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- checking a plan

def describe_error(error) -> str:
    """A short text for a Pydantic error (location: message), or the plain text of any other error."""
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())
    return str(error)[:500]


def step_error(step: Step) -> str | None:
    """None if the step's args fit the argument schema of its tool, else a text saying what is wrong.

    Arguments the schema does not know are refused too: pydantic would silently ignore them and the
    tool would then run with a different question than the one planned.
    """
    schema = ARGS_SCHEMAS[step.tool]
    unknown = sorted(set(step.args) - set(schema.model_fields))
    if unknown:
        return f"{step.tool}: unknown argument(s) {unknown}; known: {sorted(schema.model_fields)}"
    try:
        schema.model_validate(step.args)
    except ValidationError as exc:
        return f"{step.tool}: {describe_error(exc)}"
    return None


def without_why(step: Step) -> tuple[Step, bool]:
    """The step without a "why" key in its args, and whether there was one.

    The planner often repeats the reason inside args. That is harmless: the key is dropped (it is
    not an argument of any tool) and, if the step has no reason of its own, it becomes the reason.
    """
    if "why" not in step.args:
        return step, False
    args = dict(step.args)
    reason = args.pop("why")
    why = step.why or (reason if isinstance(reason, str) else "")
    return step.model_copy(update={"args": args, "why": why}), True


def salvage_steps(raw_text: str) -> list[Step]:
    """The steps of a refused plan that are well formed (tool and why), read from the raw JSON text."""
    try:
        items = json.loads(raw_text).get("steps", [])
    except (ValueError, AttributeError):
        return []
    steps = []
    for item in items if isinstance(items, list) else []:
        try:
            steps.append(Step.model_validate(item))
        except ValidationError:
            continue
    return steps


def check_plan(result: dict) -> tuple[list[Step], list[str], int]:
    """The valid steps of one planner reply, the problems found (an empty list: the plan is accepted)
    and how many steps had a "why" removed from their args.

    result is what with_structured_output(..., include_raw=True) returns: raw message, parsed Plan
    (None if the schema refused it) and parsing_error.
    """
    problems = []
    if result.get("parsed") is not None:
        steps = list(result["parsed"].steps)
    else:
        problems.append(f"the plan does not follow the schema: {describe_error(result.get('parsing_error'))}")
        steps = salvage_steps(str(result["raw"].content))
    valid, why_in_args = [], 0
    for number, step in enumerate(steps[:MAX_STEPS], start=1):
        step, found = without_why(step)
        why_in_args += found
        error = step_error(step)
        if error:
            problems.append(f"step {number}: {error}")
        else:
            valid.append(step)
    return valid, problems, why_in_args


# ---------------------------------------------------------------- the messages of the graph

def plan_message(raw: AIMessage, steps: list[Step], rejections: list[list[str]], discarded: list[dict],
                 why_in_args: int = 0, node: str = "plan", round_number: int = 0) -> AIMessage:
    """The plan as ONE model message: the steps are its tool calls, the reasons are its text.

    It keeps the planner's reasoning and its token/time metadata, and records what the bench counts:
    the rejected attempts, the retry made and the measures of the discarded first attempt.
    The repair node writes the same message with node="repair": its call ids start with "repair{n}_"
    (n = the round, from 1) and its counts are called repair_rejections / repair_retries.
    """
    prefix = "step" if node == "plan" else f"{node}{round_number}_step"
    calls = [{"name": s.tool, "args": s.args, "id": f"{prefix}_{i}", "type": "tool_call"}
             for i, s in enumerate(steps, start=1)]
    reasons = "\n".join(f"{i}. {s.tool}: {s.why}" for i, s in enumerate(steps, start=1))
    metadata = {**raw.response_metadata, "node": node, f"{node}_rejections": rejections,
                f"{node}_retries": len(discarded), "discarded_calls": discarded,
                "why_in_args": why_in_args}
    return AIMessage(content=reasons, tool_calls=calls, additional_kwargs=dict(raw.additional_kwargs),
                     response_metadata=metadata)


def run_steps(tools: dict, plan: AIMessage) -> list[ToolMessage]:
    """Run the plan's steps in order, with no model; return one tool message per step.

    A step that fails stays in the conversation as an error result and the next steps still run.
    """
    messages = []
    for call in plan.tool_calls:
        try:
            messages.append(tools[call["name"]].invoke(call))
        except Exception as exc:   # a tool bug must not stop the run: the analyst will see the error
            messages.append(ToolMessage(content=f"Error: {type(exc).__name__}: {exc}", name=call["name"],
                                        tool_call_id=call["id"], status="error"))
    return messages


def step_error_text(message: ToolMessage) -> str | None:
    """None if the step's result is a result (notes and warnings included), else the text of its error."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content)
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict) and "error" in data:
        return str(data["error"])
    if message.status == "error" or (data is None and content.lower().startswith("error")):
        return content
    return None


def executed_plans(messages: list) -> list[tuple[AIMessage, list[ToolMessage]]]:
    """Each plan or repair message of the conversation with the results of its steps, in order."""
    plans = []
    for m in messages:
        if m.type == "ai" and m.response_metadata.get("node") in ("plan", "repair"):
            plans.append((m, []))
        elif m.type == "tool" and plans and any(m.tool_call_id == c["id"] for c in plans[-1][0].tool_calls):
            plans[-1][1].append(m)
    return plans


def repair_request(messages: list, role_text: str) -> str:
    """The text of the repair request: the role, every step so far with its outcome, and the tools.

    Steps that worked take one line (tool, arguments, ok); failed steps give the whole error message.
    """
    lines = []
    for plan, results in executed_plans(messages):
        label = "Your plan" if plan.response_metadata["node"] == "plan" else "A repair of your plan"
        lines.append(f"### {label}")
        outcome = {m.tool_call_id: step_error_text(m) for m in results}
        for number, call in enumerate(plan.tool_calls, start=1):
            error = outcome.get(call["id"])
            args = json.dumps(call["args"], separators=(",", ":"))
            lines.append(f"{number}. {call['name']} {args} -> " + ("ok" if error is None else f"ERROR: {error}"))
    return f"{role_text}\n\n## What happened to your steps\n" + "\n".join(lines) + "\n\n" + tools_section()


# ---------------------------------------------------------------- the graph

def node_of(message) -> str:
    """The graph node that wrote a model message."""
    return message.response_metadata.get("node", "")


def count_calculations(messages: list) -> int:
    """How many calculate results are in the conversation."""
    return sum(1 for m in messages if m.type == "tool" and m.name == CALCULATE)


def make_rewoo_graph(analyst, seed: int, repair: bool = False):
    """Build and compile the graph: plan -> execute -> analyze <-> calculate -> END.

    With repair (graph "rewoo_repair"): after execute, if a step failed and fewer than limits.repair_rounds
    rounds were done, the repair node writes replacement steps for the failed ones, execute runs only those,
    and the analyst sees all the results. Without repair the graph is the plain ReWOO graph.
    """
    settings = analyst.domain.settings
    system = SystemMessage(content=analyst.system_text)
    tools = {t.name: t for t in analyst.tools}
    planner = make_chat_model(settings["provider"], settings["roles"]["planner"], seed).with_structured_output(
        Plan, method="json_schema", include_raw=True)
    analyst_model = make_chat_model(settings["provider"], settings["roles"]["analyst"], seed)
    with_calculate = analyst_model.bind_tools([tools[CALCULATE]])
    plan_request = analyst.roles["plan"] + "\n\n" + tools_section()
    plan_retry_request = analyst.roles.get("plan_retry", PLAN_RETRY_DEFAULT)
    analyze_request = analyst.roles["analyze"]
    empty_retry_request = analyst.roles.get("empty_retry", EMPTY_RETRY_DEFAULT)
    max_empty_retries = settings["limits"].get("empty_retries", 1)
    max_calculations = settings["limits"].get("calculate_calls", 8)
    max_repairs = settings["limits"].get("repair_rounds", 1)
    repair_role = analyst.roles.get("repair", REPAIR_DEFAULT)
    repair_retry_request = analyst.roles.get("repair_retry", REPAIR_RETRY_DEFAULT)

    def plan(state: RewooState) -> dict:
        """Ask the planner for a plan; if it is refused, ask once more with the problems found."""
        messages = [system] + state["messages"] + [HumanMessage(content=plan_request)]
        result = planner.invoke(messages)
        steps, problems, why_in_args = check_plan(result)
        rejections, discarded = [], []
        if problems:
            rejections.append(problems)
            discarded.append(call_metadata(result["raw"], "plan", "plan_retry"))
            retry = HumanMessage(content=plan_retry_request.format(problems="\n".join(problems)))
            result = planner.invoke(messages + [retry])
            steps, problems, why_in_args = check_plan(result)
            if problems:
                rejections.append(problems)
        return {"messages": [plan_message(result["raw"], steps, rejections, discarded, why_in_args)]}

    def repair_node(state: RewooState) -> dict:
        """Ask the planner for the steps that replace the failed ones; refused plans are asked once more."""
        question = next(m for m in state["messages"] if m.type == "human")
        request = HumanMessage(content=repair_request(state["messages"], repair_role))
        messages = [system, question, request]
        round_number = sum(1 for m in state["messages"] if m.type == "ai" and node_of(m) == "repair") + 1
        result = planner.invoke(messages)
        steps, problems, why_in_args = check_plan(result)
        rejections, discarded = [], []
        if problems:
            rejections.append(problems)
            discarded.append(call_metadata(result["raw"], "repair", "repair_retry"))
            retry = HumanMessage(content=repair_retry_request.format(problems="\n".join(problems)))
            result = planner.invoke(messages + [retry])
            steps, problems, why_in_args = check_plan(result)
            if problems:
                rejections.append(problems)
        return {"messages": [plan_message(result["raw"], steps, rejections, discarded, why_in_args,
                                          node="repair", round_number=round_number)]}

    def after_execute(state: RewooState) -> str:
        """Code decides: repair if a step of the last plan failed and the rounds are not used up."""
        rounds = sum(1 for m in state["messages"] if m.type == "ai" and node_of(m) == "repair")
        last_results = executed_plans(state["messages"])[-1][1]
        failed = any(step_error_text(m) is not None for m in last_results)
        return "repair" if failed and rounds < max_repairs else "analyze"

    def execute(state: RewooState) -> dict:
        """Run the plan's steps (no model)."""
        return {"messages": run_steps(tools, state["messages"][-1])}

    def analyze(state: RewooState) -> dict:
        """Ask the analyst for the answer; at the calculate limit it has no tool left.

        An empty reply is asked again (empty_retries); there is no handoff in this graph.
        """
        model = with_calculate if count_calculations(state["messages"]) < max_calculations else analyst_model
        request = HumanMessage(content=analyze_request)   # sent, not saved in the state
        reply, discarded = invoke_with_retries(model, [system] + state["messages"] + [request],
                                               empty_retry_request, max_empty_retries, "analyze")
        reply.response_metadata["node"] = "analyze"
        reply.response_metadata["discarded_calls"] = discarded
        return {"messages": [reply]}

    def after_analyze(state: RewooState) -> str:
        """Run the calculation if the analyst asked for one, otherwise stop."""
        return "calculate" if state["messages"][-1].tool_calls else END

    builder = StateGraph(RewooState)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("analyze", analyze)
    builder.add_node("calculate", ToolNode([tools[CALCULATE]]))
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    if repair:
        builder.add_node("repair", repair_node)
        builder.add_conditional_edges("execute", after_execute, ["repair", "analyze"])
        builder.add_edge("repair", "execute")
    else:
        builder.add_edge("execute", "analyze")
    builder.add_conditional_edges("analyze", after_analyze, ["calculate", END])
    builder.add_edge("calculate", "analyze")
    return builder.compile()
