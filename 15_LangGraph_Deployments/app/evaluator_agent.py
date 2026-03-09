"""An agent graph with a post-response fact check node.
After the agent responds (without tool calls), a fact check node
evaluates the factual correctness of the response.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class FactCheckResult(BaseModel):
    is_factually_correct: bool = Field(
        description="Whether the assistant's response is factually correct"
    )
    explanation: str = Field(
        description="Short explanation of the judgement"
    )


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_factcheck(state: MessagesState):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "action"

    return "fact_check"


_fact_check_prompt = ChatPromptTemplate.from_template(
    "Given a user query and the assistant's response, determine whether "
    "the response is factually correct.\n\n"
    "User Query:\n{query}\n\n"
    "Assistant Response:\n{response}"
)


def fact_check_node(state: MessagesState) -> dict:
    """Evaluate factual correctness of the assistant response."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="FACT_CHECK:END")]}

    query = state["messages"][0]
    response = state["messages"][-1]

    structured_model = get_chat_model(
        model_name="gpt-4.1-mini"
    ).with_structured_output(FactCheckResult)

    result = (_fact_check_prompt | structured_model).invoke(
        {
            "query": query.content,
            "response": response.content,
        }
    )

    verdict = "CORRECT" if result.is_factually_correct else "INCORRECT"

    return {
        "messages": [
            AIMessage(content=f"FACT_CHECK:{verdict} - {result.explanation}")
        ]
    }

def fact_check_decision(state: MessagesState):
    """Terminate on FACT_CHECK:CORRECT or loop otherwise; guard against infinite loops."""

    if any(getattr(m, "content", "") == "FACT_CHECK:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")

    if "FACT_CHECK:CORRECT" in text:
        return "end"

    return "continue"


def build_graph():
    """Build an agent graph with a summarization node."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_factcheck,
    {"action": "action", "fact_check": "fact_check"},
    )
    graph.add_conditional_edges(
        "fact_check",
        fact_check_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
