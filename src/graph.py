"""
智能面试模拟与反馈系统 — LangGraph 状态图 (Web 交互版)

核心改动：用 interrupt() 替代 LLM 模拟回答，支持真实用户逐步输入。

流程：
  START → supervisor → question_design_flow (并行出题) → supervisor
  → human_answer (interrupt 暂停等用户输入) → supervisor
  → evaluation_flow (并行评估) → supervisor
  → [追问回路 or 报告] → finish
"""

from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache
from statistics import mean
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, Send, interrupt

from src.agents import (
    coach_report,
    design_question,
    evaluate_answer,
    generate_follow_up,
    supervisor_decide,
)
from src.models import (
    CandidateAnswer,
    InterviewAgentState,
    InterviewQuestion,
    SingleEvaluationState,
    SingleQuestionState,
)
from src.utils import get_checkpoint_db_path

QUESTION_CATEGORIES: list[Literal["technical", "project_deep_dive", "behavioral"]] = [
    "technical", "project_deep_dive", "behavioral"
]


@lru_cache(maxsize=1)
def build_sqlite_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(get_checkpoint_db_path(), check_same_thread=False)
    return SqliteSaver(conn)


# ── 辅助函数 ────────────────────────────────────────────────

def _get_active_evaluations(state: InterviewAgentState) -> list[dict]:
    current_round = state.get("interview_round", 0)
    return [
        e for e in state.get("evaluations", [])
        if e.get("answer_round", 0) == current_round
    ]


def _get_active_questions(state: InterviewAgentState) -> list[dict]:
    """只返回当前轮次需要回答的题目"""
    current_round = state.get("interview_round", 0)
    questions = state.get("questions", [])
    active = [q for q in questions if q.get("answer_round", current_round) == current_round]
    return active if active else questions


def _build_supervisor_payload(state: InterviewAgentState) -> str:
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    evaluations = _get_active_evaluations(state)
    scores = [e["score"] for e in evaluations]
    avg_score = round(mean(scores), 2) if scores else 0
    poor_count = sum(1 for e in evaluations if e["verdict"] in ("poor", "needs_improvement"))

    return f"""
求职目标：{state['user_goal']}

状态摘要：
- 题目数：{len(questions)}
- 回答数：{len(answers)}
- 评估数：{len(evaluations)}
- 平均分：{avg_score}
- 不及格数：{poor_count}
- 当前轮次：{state.get('interview_round', 0)} / {state.get('max_rounds', 2)}
- 已有报告：{bool(state.get('final_report'))}

请决定下一步。"""


def _fallback_supervisor_step(state: InterviewAgentState) -> tuple[str, str]:
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    evaluations = _get_active_evaluations(state)
    scores = [e["score"] for e in evaluations]
    avg_score = mean(scores) if scores else 0
    poor_count = sum(1 for e in evaluations if e["verdict"] in ("poor", "needs_improvement"))

    if state.get("final_report"):
        return "finish", "报告已生成。"
    if not questions:
        return "design_questions", "先出题。"
    if not answers:
        return "simulate_answers", "等用户回答。"  # web 版中 simulate_answers 即 human_answer
    if not evaluations:
        return "evaluate_answers", "需要评估。"
    if poor_count > 0 and state.get("interview_round", 0) < state.get("max_rounds", 2):
        return "follow_up", f"{poor_count} 道题不及格，追问。"
    return "finish", "评估完成。"


def _is_step_allowed(state: InterviewAgentState, step: str) -> bool:
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    evaluations = _get_active_evaluations(state)
    has_report = bool(state.get("final_report"))
    rounds_left = state.get("interview_round", 0) < state.get("max_rounds", 2)

    if step == "design_questions":
        return not questions
    if step == "simulate_answers":
        return bool(questions) and not answers
    if step == "evaluate_answers":
        return bool(answers) and not evaluations
    if step == "follow_up":
        poor = any(e["verdict"] in ("poor", "needs_improvement") for e in evaluations)
        return bool(evaluations) and poor and rounds_left
    if step == "coach":
        return bool(evaluations)
    if step == "finish":
        return has_report
    return False


# ── Supervisor 节点 ─────────────────────────────────────────

def supervisor_node(
    state: InterviewAgentState,
) -> Command[Literal[
    "question_design_flow", "human_answer", "evaluation_flow",
    "follow_up_generator", "feedback_coach", "finish_node"
]]:
    fallback_step, fallback_reason = _fallback_supervisor_step(state)

    try:
        decision = supervisor_decide(_build_supervisor_payload(state))
        next_step = decision.get("next_step", fallback_step)
        reason = decision.get("reason", fallback_reason)
    except Exception:
        next_step = fallback_step
        reason = fallback_reason

    allowed = {"design_questions", "simulate_answers", "evaluate_answers", "follow_up", "coach", "finish"}
    if next_step not in allowed:
        next_step = fallback_step
        reason = fallback_reason

    if not _is_step_allowed(state, next_step):
        next_step = fallback_step
        reason = fallback_reason

    if state.get("final_report"):
        next_step = "finish"

    if next_step == "design_questions":
        goto: list = [
            Send("question_design_flow", {
                "category": cat,
                "goal_context": state["user_goal"],
                "resume_context": state["resume_text"],
            })
            for cat in QUESTION_CATEGORIES
        ]
    elif next_step == "simulate_answers":
        goto = "human_answer"   # web 版：跳转到人类回答节点
    elif next_step == "evaluate_answers":
        questions = state.get("questions", [])
        answers = state.get("answers", [])
        answer_map = {a["question_id"]: a for a in answers}
        question_map = {q["question_id"]: q for q in questions}
        eval_inputs: list = []
        for aid, answer in answer_map.items():
            q = question_map.get(aid)
            if q:
                eval_inputs.append(Send("evaluation_flow", {
                    "question": q,
                    "answer": answer,
                    "goal_context": state["user_goal"],
                    "resume_context": state["resume_text"],
                }))
        goto = eval_inputs
    elif next_step == "follow_up":
        goto = "follow_up_generator"
    elif next_step == "coach":
        goto = "feedback_coach"
    else:
        goto = "finish_node"

    return Command(
        update={
            "next_step": next_step,
            "strategy_reason": reason,
            "messages": [AIMessage(content=f"Supervisor 决策：{next_step}。{reason}")],
        },
        goto=goto,
    )


# ── 出题子图 ────────────────────────────────────────────────

def design_question_node(state: SingleQuestionState) -> dict:
    result = design_question(
        category=state["category"],
        user_goal=state["goal_context"],
        resume_text=state["resume_context"],
    )
    question: InterviewQuestion = {
        "question_id": f"q-{state['category']}-{uuid.uuid4().hex[:6]}",
        "category": state["category"],
        "question_text": result.get("question_text", ""),
        "expected_points": result.get("expected_points", []),
        "difficulty": result.get("difficulty", "medium"),
    }
    return {"questions": [question]}


def build_question_design_subgraph():
    graph = StateGraph(SingleQuestionState)
    graph.add_node("design_question_node", design_question_node)
    graph.add_edge(START, "design_question_node")
    graph.add_edge("design_question_node", END)
    return graph.compile()


# ── ⭐ 人类回答节点（核心改动：interrupt 暂停等待用户输入） ──

def human_answer_node(
    state: InterviewAgentState,
) -> Command[Literal["supervisor"]]:
    questions = _get_active_questions(state)
    current_round = state.get("interview_round", 0)

    # 准备给前端的数据
    question_data = [
        {
            "question_id": q["question_id"],
            "category": q["category"],
            "question_text": q["question_text"],
            "difficulty": q.get("difficulty", "medium"),
        }
        for q in questions
    ]

    # ⬇︎ 暂停图执行，返回题目给前端。用户提交回答后图从这里恢复。
    user_input = interrupt({
        "type": "wait_for_answer",
        "questions": question_data,
        "round": current_round,
        "message": f"请回答 {len(question_data)} 道面试题",
    })

    # 用户提交的回答格式：{"answers": [{"question_id": "...", "answer_text": "..."}, ...]}
    raw_answers = user_input.get("answers", []) if isinstance(user_input, dict) else []

    answers: list[CandidateAnswer] = [
        {
            "question_id": a["question_id"],
            "answer_text": a.get("answer_text", ""),
            "answer_round": current_round,
        }
        for a in raw_answers
    ]

    return Command(
        update={
            "answers": answers,
            "messages": [AIMessage(content=f"收到 {len(answers)} 份回答（第 {current_round + 1} 轮）")],
        },
        goto="supervisor",
    )


# ── 评估子图 ────────────────────────────────────────────────

def evaluate_answer_node(state: SingleEvaluationState) -> dict:
    result = evaluate_answer(
        question=state["question"],
        answer_text=state["answer"]["answer_text"],
        user_goal=state["goal_context"],
        resume_text=state["resume_context"],
    )
    evaluation = {
        "question_id": state["question"]["question_id"],
        "answer_round": state["answer"].get("answer_round", 0),
        "score": result.get("score", 0),
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
        "verdict": result.get("verdict", "good"),
        "follow_up_question": result.get("follow_up_question", ""),
    }
    return {"evaluations": [evaluation]}


def build_evaluation_subgraph():
    graph = StateGraph(SingleEvaluationState)
    graph.add_node("evaluate_answer_node", evaluate_answer_node)
    graph.add_edge(START, "evaluate_answer_node")
    graph.add_edge("evaluate_answer_node", END)
    return graph.compile()


# ── 追问生成节点 ────────────────────────────────────────────

def follow_up_generator_node(
    state: InterviewAgentState,
) -> Command[Literal["supervisor"]]:
    evaluations = _get_active_evaluations(state)
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    question_map = {q["question_id"]: q for q in questions}
    answer_map = {a["question_id"]: a for a in answers}

    new_questions: list = []
    for eval_result in evaluations:
        if eval_result["verdict"] in ("poor", "needs_improvement"):
            q = question_map.get(eval_result["question_id"], {})
            a = answer_map.get(eval_result["question_id"], {})
            follow_up = generate_follow_up(
                question=q,
                evaluation=eval_result,
                answer_text=a.get("answer_text", ""),
            )
            next_round = state.get("interview_round", 0) + 1
            new_questions.append({
                "question_id": f"fu-{q.get('question_id', 'unknown')}-r{next_round}",
                "category": q.get("category", "technical"),
                "question_text": follow_up.get("question_text", ""),
                "expected_points": follow_up.get("expected_points", []),
                "difficulty": follow_up.get("difficulty", "medium"),
                "answer_round": next_round,
            })

    return Command(
        update={
            "interview_round": state.get("interview_round", 0) + 1,
            "questions": new_questions,
            "answers": [],
            "messages": [AIMessage(content=f"生成 {len(new_questions)} 道追问，进入第 {state.get('interview_round', 0) + 1} 轮。")],
        },
        goto="supervisor",
    )


# ── 面试教练节点 ────────────────────────────────────────────

def feedback_coach_node(
    state: InterviewAgentState,
) -> Command[Literal["supervisor"]]:
    evaluations = _get_active_evaluations(state)
    questions = state.get("questions", [])
    answers = state.get("answers", [])

    report = coach_report(
        user_goal=state["user_goal"],
        questions=questions,
        answers=answers,
        evaluations=evaluations,
    )

    return Command(
        update={
            "final_report": report,
            "messages": [AIMessage(content="面试教练已生成反馈报告。")],
        },
        goto="supervisor",
    )


# ── 结束节点 ────────────────────────────────────────────────

def finish_node(state: InterviewAgentState) -> dict:
    return {"messages": [AIMessage(content="流程结束。")]}


# ── 构建主图 ────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(InterviewAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("question_design_flow", build_question_design_subgraph())
    graph.add_node("human_answer", human_answer_node)
    graph.add_node("evaluation_flow", build_evaluation_subgraph())
    graph.add_node("follow_up_generator", follow_up_generator_node)
    graph.add_node("feedback_coach", feedback_coach_node)
    graph.add_node("finish_node", finish_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("question_design_flow", "supervisor")
    graph.add_edge("human_answer", "supervisor")
    graph.add_edge("evaluation_flow", "supervisor")
    graph.add_edge("follow_up_generator", "supervisor")
    graph.add_edge("feedback_coach", "supervisor")
    graph.add_edge("finish_node", END)

    return graph.compile(checkpointer=build_sqlite_checkpointer())
