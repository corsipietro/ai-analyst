"""The analyst graph: an explorer collects data with the tools, then an interpreter writes the answer.

Ported from analyst/fast_graph.py (v1, 29/10/2026), without any domain code:
1. the instructions are the generic method + the system context (context.md) + the database
   description, read once by the code, so the model does not spend calls asking for them;
2. explorer (no reasoning, deterministic, short token cap) calls tools until it replies DATA COLLECTED;
   any text it writes instead of a tool call is replaced by DATA COLLECTED (no draft answers);
3. interpreter (reasoning on) writes the final answer and may still ask for data;
   if it replies with nothing (no text, no tool call), it is asked again, at most
   limits.empty_retries times (bench of 29/10 evening: 4 empty answers out of 18, each after a
   reasoning that planned another tool call but did not make it);
4. if it is still empty, the work is handed back to the explorer (at most limits.handoffs times)
   with the plan the interpreter wrote in its reasoning: the explorer, without reasoning, makes
   tool calls reliably (second test of 29/10: a second request alone fixed 1 empty answer of 4).
Model, sampling and token caps of each role come from the settings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .config import Domain, check_fingerprint, load_domain, load_prompt
from .data_access import DataSource
from .llm import make_chat_model
from .tool_schemas import make_langchain_tools
from .tools import Toolbox

DATA_COLLECTED = "DATA COLLECTED"
EMPTY_RETRY_DEFAULT = ("Your last reply was empty. Either call a tool to get the data you still need, "
                       "or write the final answer now.")
EMPTY_RETRY_NOTES_DEFAULT = ("Your last reply was empty. These are the last notes of your reasoning:\n{notes}\n"
                             "Using these notes and the data above, write the final answer now "
                             "(or call a tool if you still need one).")
EMPTY_RETRY_MODES = ("same", "no_think")
# describe_database is already in the instructions: the graph does not offer it as a tool.
GRAPH_TOOLS = ["query_results", "summarize_results", "describe_trends", "compare_groups", "check_quality",
               "calculate"]


# The tools whose results are data (describe_database and check_quality describe the structure).
READING_TOOLS = ("summarize_results", "describe_trends", "compare_groups", "query_results")   # a plan without them is repaired
ANSWER_TOOLS = READING_TOOLS + ("check_quality",)   # no header if one of these worked: missing-value counts are data (3/11)
NO_DATA_HEADER = ("**Data not retrieved.** Every step that reads the data failed, so nothing below comes from the "
                  "data: it rests only on the question and the context.\n\n")

HANDOFF_MARK = "[HANDOFF]"
PLAN_CHARS = 1500   # tail of the interpreter's reasoning passed to the explorer


class AnalystState(MessagesState):
    """The messages, which role spoke last ('explore', 'interpret' or 'handoff'), handoffs made."""

    phase: str
    handoffs: int


@dataclass
class Analyst:
    """Everything needed to answer questions about one domain: data, tools, instructions."""

    domain: Domain
    source: DataSource
    toolbox: Toolbox
    tools: list
    system_text: str
    reference: str        # the database description given in the instructions (a source for the checker)
    roles: dict           # role requests from prompts/roles_<version>.yaml

    def close(self) -> None:
        """Close the data connection."""
        self.source.close()


def build_system_text(method: str, context: str, reference: str, answer_language: str) -> str:
    """Join the generic method, the system context, the database description and the answer language.

    The language is a run setting, not part of the method, so it is added here at the end.
    """
    return (method.strip()
            + "\n\n## System context (provided by the user)\n" + context.strip()
            + "\n\n## Database description (already read for you with describe_database)\n" + reference
            + "\n\n## Output language\nAlways answer in " + answer_language
            + ", whatever the language of the question.")


def open_analyst(domain_name: str, overrides: dict | None = None) -> Analyst:
    """Load a domain, check the data fingerprint, open the data read-only and prepare the tools.

    overrides: nested settings that replace those of settings.yaml and of the domain (bench --set).
    """
    domain = load_domain(domain_name, overrides=overrides)
    check_fingerprint(domain.data)
    source = DataSource(domain.data)
    toolbox = Toolbox(source, domain.settings)
    description = toolbox.describe_database()
    if "error" in description:
        source.close()
        raise RuntimeError(f"describe_database failed: {description['error']}")
    reference = json.dumps(description, ensure_ascii=False)
    method, roles = load_prompt(domain.settings["prompt"])
    tools = [t for t in make_langchain_tools(toolbox) if t.name in GRAPH_TOOLS]
    return Analyst(domain=domain, source=source, toolbox=toolbox, tools=tools,
                   system_text=build_system_text(method, domain.context, reference,
                                                 domain.settings["answer_language"]),
                   reference=reference, roles=roles)


def without_draft(reply: AIMessage) -> AIMessage:
    """Replace any text the explorer wrote instead of a tool call with DATA COLLECTED.

    The token counts stay in response_metadata, so the log still shows whether it disobeyed.
    """
    if reply.tool_calls:
        return reply
    return AIMessage(content=DATA_COLLECTED, response_metadata=reply.response_metadata, id=reply.id)


def step_error_text(message: ToolMessage) -> str | None:
    """None if the tool message holds a result (notes and warnings included), else the text of its error."""
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


def data_retrieved(messages: list, tools: tuple = READING_TOOLS) -> bool:
    """True if at least one of the tools (default READING_TOOLS) gave a result (not an error) in the conversation."""
    return any(m.type == "tool" and m.name in tools and step_error_text(m) is None for m in messages)


def with_no_data_header(reply: AIMessage, messages: list) -> AIMessage:
    """The final reply with NO_DATA_HEADER in front if no tool of ANSWER_TOOLS gave a result (a reply that asks for a tool, or
    an empty one, is returned as it is)."""
    if reply.tool_calls or is_empty(reply) or data_retrieved(messages, ANSWER_TOOLS):
        return reply
    reply.content = NO_DATA_HEADER + str(reply.content)
    return reply


def is_empty(reply: AIMessage) -> bool:
    """True if a model reply has neither a tool call nor any visible text."""
    return not reply.tool_calls and not str(reply.content).strip()


def call_metadata(reply: AIMessage, node: str, reason: str, reasoning: bool = True) -> dict:
    """The measures of one discarded model call: node, why it was discarded, tokens, durations.

    reason is "empty_retry" (the reply was empty and the interpreter was asked again) or
    "handoff" (it was empty and the work went back to the explorer). done_reason is how the model
    stopped ("" if unknown, "length" if it ran out of tokens); reasoning says whether the model of
    the call had reasoning on.
    """
    meta = reply.response_metadata
    return {"node": node, "reason": reason, **{k: meta.get(k, 0) or 0 for k in
                             ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration")},
            "done_reason": meta.get("done_reason", "") or "", "reasoning": reasoning}


def empty_retry_mode(settings: dict) -> str:
    """The setting limits.empty_retry_mode ("same" by default); an unknown value is a ValueError."""
    mode = settings["limits"].get("empty_retry_mode", "same")
    if mode not in EMPTY_RETRY_MODES:
        raise ValueError(f"limits.empty_retry_mode must be one of {list(EMPTY_RETRY_MODES)}, got {mode!r}")
    return mode


def empty_retry_notes(reply: AIMessage, template: str) -> str:
    """The retry request with the tail (PLAN_CHARS) of the discarded reply's reasoning in place of {notes}."""
    reasoning = str(reply.additional_kwargs.get("reasoning_content", "")).strip()
    notes = reasoning[-PLAN_CHARS:] if reasoning else "(no notes written)"
    return template.replace("{notes}", notes)


def invoke_with_retries(model, messages: list, retry_request: str, max_retries: int, node: str,
                        fallback_model=None, notes_request: str | None = None, reasoning: bool = True):
    """Call the model; while the reply is empty, ask again (at most max_retries times) with one more request.

    Returns the last reply and the measures of the empty replies that were thrown away
    (reason "empty_retry"). Shared by the interpreter of this graph and the analyst of rewoo.py.
    With fallback_model (a model of the same role with reasoning off) the retries go to it, with
    notes_request (placeholder {notes}: the tail of the discarded reasoning) instead of retry_request;
    a reply it gives has response_metadata["empty_retry_mode"] = "no_think". reasoning says whether
    `model` has reasoning on (recorded in the measures).
    """
    reply = model.invoke(messages)
    discarded = []
    while is_empty(reply) and len(discarded) < max_retries:
        discarded.append(call_metadata(reply, node, "empty_retry", reasoning))
        if fallback_model is not None:
            request = empty_retry_notes(reply, notes_request or EMPTY_RETRY_NOTES_DEFAULT)
            model, reasoning = fallback_model, False
            messages = messages + [HumanMessage(content=request)]
            reply = model.invoke(messages)
            reply.response_metadata["empty_retry_mode"] = "no_think"
        else:
            messages = messages + [HumanMessage(content=retry_request)]
            reply = model.invoke(messages)
    return reply, discarded


def handoff_message(reply: AIMessage, discarded: list[dict]) -> HumanMessage:
    """The message that hands the work back to the explorer, with the end of the interpreter's plan.

    It is saved in the conversation, so it appears in the transcript and the bench can count it.
    The empty replies that led to the handoff are discarded, but their measures are kept in
    additional_kwargs["discarded_calls"], so the bench still counts those model calls.
    """
    reasoning = str(reply.additional_kwargs.get("reasoning_content", "")).strip()
    plan = reasoning[-PLAN_CHARS:] if reasoning else "(no plan written)"
    return HumanMessage(content=f"{HANDOFF_MARK} The analyst writing the answer needs more data before "
                                f"answering. The end of its plan:\n{plan}\n"
                                "Collect the data this plan asks for, with the tools.",
                        additional_kwargs={"discarded_calls": discarded})


def make_graph(analyst: Analyst, seed: int):
    """Build and compile the graph: explore <-> tools, then interpret (<-> tools) -> END."""
    settings = analyst.domain.settings
    system = SystemMessage(content=analyst.system_text)
    explorer = make_chat_model(settings["provider"], settings["roles"]["explorer"], seed).bind_tools(analyst.tools)
    interpreter = make_chat_model(settings["provider"], settings["roles"]["interpreter"], seed).bind_tools(analyst.tools)
    explore_request = analyst.roles["explore"].replace("{DATA_COLLECTED}", DATA_COLLECTED)
    interpret_request = analyst.roles["interpret"]
    empty_retry_request = analyst.roles.get("empty_retry", EMPTY_RETRY_DEFAULT)
    max_empty_retries = settings["limits"].get("empty_retries", 1)
    max_handoffs = settings["limits"].get("handoffs", 1)
    no_think = empty_retry_mode(settings) == "no_think"
    interpreter_reasoning = bool(settings["roles"]["interpreter"].get("reasoning", False))
    retry_model = (make_chat_model(settings["provider"], {**settings["roles"]["interpreter"], "reasoning": False},
                                   seed).bind_tools(analyst.tools) if no_think else None)
    notes_request = analyst.roles.get("empty_retry_notes", EMPTY_RETRY_NOTES_DEFAULT)

    def explore(state: AnalystState) -> dict:
        """Ask the explorer for the next tool call, or for DATA COLLECTED."""
        request = HumanMessage(content=explore_request)   # sent, not saved in the state
        reply = explorer.invoke([system] + state["messages"] + [request])
        reply.response_metadata["node"] = "explore"   # kept by without_draft: the bench measures per node
        return {"messages": [without_draft(reply)], "phase": "explore"}

    def interpret(state: AnalystState) -> dict:
        """Ask the interpreter (with reasoning) for the final answer, or for more data.

        An empty reply (no text and no tool call) is not kept: the interpreter is asked again,
        with one more request appended, at most max_empty_retries times. The measures of every
        discarded call, with the reason, go in response_metadata["discarded_calls"] of the reply
        that is kept (or in the handoff message): they cost time and tokens, and the bench counts
        the retries from them.
        """
        request = HumanMessage(content=interpret_request)   # sent, not saved in the state
        reply, discarded = invoke_with_retries(interpreter, [system] + state["messages"] + [request],
                                               empty_retry_request, max_empty_retries, "interpret",
                                               fallback_model=retry_model, notes_request=notes_request,
                                               reasoning=interpreter_reasoning)
        reply.response_metadata["node"] = "interpret"
        reply = with_no_data_header(reply, state["messages"])
        if is_empty(reply) and state.get("handoffs", 0) < max_handoffs:
            last_reasoning = interpreter_reasoning and reply.response_metadata.get("empty_retry_mode") != "no_think"
            discarded.append(call_metadata(reply, "interpret", "handoff", last_reasoning))
            return {"messages": [handoff_message(reply, discarded)], "phase": "handoff",
                    "handoffs": state.get("handoffs", 0) + 1}
        reply.response_metadata["discarded_calls"] = discarded
        return {"messages": [reply], "phase": "interpret"}

    def after_explore(state: AnalystState) -> str:
        """Go to the tools if the explorer asked for one, otherwise to the interpreter."""
        return "tools" if state["messages"][-1].tool_calls else "interpret"

    def after_interpret(state: AnalystState) -> str:
        """Tools if the interpreter asked for one, back to the explorer after a handoff, otherwise stop."""
        if state.get("phase") == "handoff":
            return "explore"
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AnalystState) -> str:
        """Give the tool results back to the role that asked for them."""
        return state.get("phase", "explore")

    builder = StateGraph(AnalystState)
    builder.add_node("explore", explore)
    builder.add_node("interpret", interpret)
    builder.add_node("tools", ToolNode(analyst.tools))
    builder.add_edge(START, "explore")
    builder.add_conditional_edges("explore", after_explore, ["tools", "interpret"])
    builder.add_conditional_edges("interpret", after_interpret, ["tools", "explore", END])
    builder.add_conditional_edges("tools", after_tools, ["explore", "interpret"])
    return builder.compile()
