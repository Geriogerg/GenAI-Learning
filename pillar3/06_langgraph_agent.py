"""
Step 6 - the same agent in LangGraph, with a human approval step.

LangGraph models an agent as a graph:
  - a STATE shared by all nodes (here: the list of messages)
  - NODES that read the state and return updates (call the LLM, run tools...)
  - EDGES, possibly conditional, that decide which node runs next

    START -> agent --(tool calls?)--> tools -> agent -> ... --(no tool calls)--> END

What you get compared to our hand-written loop:
  - checkpointing: the state is saved after every node (here in memory, in prod in a DB),
    so a conversation can be resumed later, by thread_id
  - interrupt(): pause the graph in the middle of a node, wait for a human, resume
  - the flow is explicit, you can draw it, and add nodes (a reviewer, a router...) without
    turning the while loop into spaghetti

Usage: python 06_langgraph_agent.py
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command, interrupt

from agent_utils import MODEL
from registry_tools import FUNCTIONS, SYSTEM_PROMPT, flag_for_retraining

# same plain functions as before, wrapped as LangChain tools
lc_tools = [StructuredTool.from_function(f) for f in FUNCTIONS]
tools_by_name = {t.name: t for t in lc_tools}
NEEDS_APPROVAL = {flag_for_retraining.__name__}

llm = ChatGroq(model=MODEL, temperature=0.2, max_tokens=2000,
               reasoning_effort="low" if "gpt-oss" in MODEL else None)
llm_with_tools = llm.bind_tools(lc_tools)


# --- nodes -------------------------------------------------------------------

def agent_node(state: MessagesState):
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def tools_node(state: MessagesState):
    """Runs the tool calls of the last AI message. Pauses for approval on sensitive tools."""
    outputs = []
    for call in state["messages"][-1].tool_calls:
        if call["name"] in NEEDS_APPROVAL:
            # The graph stops here and invoke() returns. When resumed with Command(resume=...),
            # the node runs again from the top and interrupt() returns the human's answer.
            # (So anything before interrupt() runs twice: keep side effects after it.)
            decision = interrupt({"tool": call["name"], "args": call["args"]})
            if decision != "yes":
                outputs.append(ToolMessage("The user refused this action. Don't retry it.",
                                           tool_call_id=call["id"]))
                continue
        try:
            result = tools_by_name[call["name"]].invoke(call["args"])
            content = result if isinstance(result, str) else json.dumps(result)
        except Exception as e:  # give the error to the model, same idea as in step 2
            content = f"Error: {type(e).__name__}: {e}"
        outputs.append(ToolMessage(content, tool_call_id=call["id"]))
    return {"messages": outputs}


# --- graph -------------------------------------------------------------------

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" if tool calls, else END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


def run(app, thread_id, text):
    """Send a message on a thread and handle approvals until the graph finishes."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}  # ~ step limit
    result = app.invoke({"messages": [HumanMessage(text)]}, config)

    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"\n  >>> approval needed: {request['tool']}({request['args']})")
        answer = "yes" if input("  >>> approve? [y/N] ").strip().lower() in ("y", "yes") else "no"
        result = app.invoke(Command(resume=answer), config)

    return result["messages"][-1].content


def main():
    app = build_graph()
    print("The graph, in Mermaid syntax (paste it on mermaid.live to see it):\n")
    print(app.get_graph().draw_mermaid())

    thread = "briefing-1"
    for text in [
        "Which production models are drifting?",
        "Queue the worst one for retraining, high priority.",   # -> approval prompt
        "And unet_coating too.",                                # -> approval, then a budget error
        "What did I ask you to queue so far?",                  # the checkpointer kept the thread
    ]:
        print("\n" + "=" * 80)
        print(f"You: {text}")
        print(f"Agent: {run(app, thread, text)}")

    # Same thread_id = same conversation. A new thread_id starts from scratch:
    print("\n" + "=" * 80)
    print("You (new thread): What did I ask you to queue so far?")
    print(f"Agent: {run(app, 'briefing-2', 'What did I ask you to queue so far?')}")


if __name__ == "__main__":
    main()

# Notes:
#  - langchain also has a ready-made create_agent() that builds this exact graph in one line.
#    Build it by hand once (like here) so you know what's inside, then use the shortcut.
#  - InMemorySaver dies with the script. langgraph-checkpoint-sqlite / -postgres store
#    threads in a DB, which is how you'd resume an agent after a restart or on another server.
#  - Exercise: add a "reviewer" node between agent and END that checks the final answer
#    quotes numbers from the tool results, and sends it back to "agent" if not.
