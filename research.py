from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

class ResearchQuestion(BaseModel):
    id: int
    question: str
    goal: str

class Evidence(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None

class ResearchPlan(BaseModel):
    topic: str
    questions: List[ResearchQuestion]

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = []
    reason: str

class State(TypedDict):
    topic: str
    as_of: str

    mode: str
    needs_research: bool
    queries: List[str]

    plan: Optional[ResearchPlan]
    evidence: List[Evidence]

    notes: Annotated[List[str], operator.add]
    final_report: str


llm = ChatOpenAI(model="gpt-5.1")

ROUTER_SYSTEM = """
You are a research router.

Decide if the topic requires web research.

Modes:
- closed_book: foundational knowledge
- hybrid: concepts + recent tools/models
- open_book: latest events, papers, releases
"""

def router_node(state: State) -> dict:
    router = llm.with_structured_output(RouterDecision)
    decision = router.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs of: {state['as_of']}")
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "planner"

def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        return tool.invoke({"query": query}) or []
    except Exception:
        return []


RESEARCH_SYSTEM = """Extract clean, factual evidence.
Only include items with URLs.
Do not hallucinate dates or sources.
"""

def research_node(state: State) -> dict:
    raw = []
    for q in state.get("queries", []):
        raw.extend(tavily_search(q))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(List[Evidence])
    evidence = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=str(raw))
        ]
    )

    return {"evidence": evidence}

PLANNER_SYSTEM = """
You are a senior research lead.

Break the topic into 4–7 research questions.
Questions must be non-overlapping and precise.
"""

def planner_node(state: State) -> dict:
    planner = llm.with_structured_output(ResearchPlan)
    plan = planner.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}")
        ]
    )
    return {"plan": plan}

def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "question": q.model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
                "mode": state["mode"]
            }
        )
        for q in state["plan"].questions
    ]

WORKER_SYSTEM = """
You are a research analyst.

Answer ONE research question.
If mode=open_book, only use provided evidence URLs.
Cite sources using markdown links.
"""

def worker_node(payload: dict) -> dict:
    q = payload["question"]
    evidence = payload.get("evidence", [])

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {q['question']}\n"
                    f"Goal: {q['goal']}\n\n"
                    f"Evidence:\n{evidence}"
                )
            )
        ]
    ).content

    return {"notes": [response]}

SYNTH_SYSTEM = """
Synthesize a structured research report:
- Executive summary
- Key findings
- Risks / gaps
- Conclusion
"""

def synth_node(state: State) -> dict:
    response = llm.invoke(
        [
            SystemMessage(content=SYNTH_SYSTEM),
            HumanMessage(content="\n\n".join(state["notes"]))
        ]
    )
    return {"final_report": response.content}

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("planner", planner_node)
g.add_node("worker", worker_node)
g.add_node("synth", synth_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {
    "research": "research",
    "planner": "planner"
})
g.add_edge("research", "planner")

g.add_conditional_edges("planner", fanout, ["worker"])
g.add_edge("worker", "synth")
g.add_edge("synth", END)

app = g.compile()

if __name__ == "__main__":
    result = app.invoke({
        "topic": "langgraph",
        "as_of": date.today().isoformat(),
        "notes": [],
        "evidence": []
    })

    print(result["final_report"])


