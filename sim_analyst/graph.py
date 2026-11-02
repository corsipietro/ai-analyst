"""The analyst graph: an explorer collects data with the tools, then an interpreter writes the answer.

How it works:
1. the instructions are the generic method + the system context (context.md) + the database
   description, read once by the code, so the model does not spend calls asking for them;
2. explorer (by default: no reasoning, deterministic, short token cap) calls tools until it replies DATA COLLECTED;
   any text it writes instead of a tool call is replaced by DATA COLLECTED (no draft answers);
3. interpreter (reasoning on) writes the final answer and may still ask for data;
   if it replies with nothing (no text, no tool call), it is asked again, at most
   limits.empty_retries times (an empty reply usually follows a reasoning that planned another
   tool call but did not make it);
4. if it is still empty, the work is handed back to the explorer (at most limits.handoffs times)
   with the plan the interpreter wrote in its reasoning: the explorer, without reasoning, makes
   tool calls reliably.
Model, sampling and token caps of each role come from the settings (episode 1 ran the explorer
with reasoning on: see results/episode1/reproduce.sh).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
# describe_database is already in the instructions: the graph does not offer it as a tool.
GRAPH_TOOLS = ["query_results", "summarize_results", "describe_trends", "compare_groups", "check_quality",
               "calculate"]


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


def is_empty(reply: AIMessage) -> bool:
    """True if a model reply has neither a tool call nor any visible text."""
    return not reply.tool_calls and not str(reply.content).strip()


def call_metadata(reply: AIMessage, node: str, reason: str) -> dict:
    """The measures of one discarded model call: node, why it was discarded, tokens, durations.

    reason is "empty_retry" (the reply was empty and the interpreter was asked again) or
    "handoff" (it was empty and the work went back to the explorer).
    """
    meta = reply.response_metadata
    return {"node": node, "reason": reason, **{k: meta.get(k, 0) or 0 for k in
                             ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration")}}


def invoke_with_retries(model, messages: list, retry_request: str, max_retries: int, node: str):
    """Call the model; while the reply is empty, ask again (at most max_retries times) with one more request.

    Returns the last reply and the measures of the empty replies that were thrown away
    (reason "empty_retry"). Shared by the interpreter of this graph and the analyst of rewoo.py.
    """
    reply = model.invoke(messages)
    discarded = []
    while is_empty(reply) and len(discarded) < max_retries:
        discarded.append(call_metadata(reply, node, "empty_retry"))
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
                                               empty_retry_request, max_empty_retries, "interpret")
        reply.response_metadata["node"] = "interpret"
        if is_empty(reply) and state.get("handoffs", 0) < max_handoffs:
            discarded.append(call_metadata(reply, "interpret", "handoff"))
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
