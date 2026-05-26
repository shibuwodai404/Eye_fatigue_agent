"""LangGraph 主图：顺序编排 8 个 agent，消融开关在节点内部短路。

State 一直累加，每个节点返回新 state；图本身不做条件分支，
因为 "禁用某节点" 已在 agent 内部用 settings.ablation.* 处理。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents import (
    case_library,
    consistency,
    diagnosis,
    feature_aggregator,
    intake,
    report,
    retrieval,
    vision,
)
from src.state import GraphState


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("intake", intake.run)
    g.add_node("vision", vision.run)
    g.add_node("aggregator", feature_aggregator.run)
    g.add_node("retrieval", retrieval.run)
    g.add_node("diagnosis", diagnosis.run)
    g.add_node("consistency", consistency.run)
    g.add_node("case_library", case_library.run)
    g.add_node("report", report.run)

    g.add_edge(START, "intake")
    g.add_edge("intake", "vision")
    g.add_edge("vision", "aggregator")
    g.add_edge("aggregator", "retrieval")
    g.add_edge("retrieval", "diagnosis")
    g.add_edge("diagnosis", "consistency")
    g.add_edge("consistency", "case_library")
    g.add_edge("case_library", "report")
    g.add_edge("report", END)

    return g.compile()


# 模块级单例，避免每次 invoke 都重新编译
compiled_graph = build_graph()
