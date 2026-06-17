from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages


# ── 面试题定义 ──────────────────────────────────────────────

class InterviewQuestion(TypedDict):
    """单道面试题"""
    question_id: str
    category: Literal["technical", "project_deep_dive", "behavioral"]
    question_text: str
    expected_points: list[str]          # 期望候选人覆盖的要点
    difficulty: Literal["basic", "medium", "hard"]


# ── 候选人回答 ──────────────────────────────────────────────

class CandidateAnswer(TypedDict):
    """候选人对某道题的回答"""
    question_id: str
    answer_text: str
    answer_round: int                   # 第几轮回答（追问会产生多轮）


# ── 评估结果 ────────────────────────────────────────────────

class EvaluationResult(TypedDict):
    """对单道题回答的评估"""
    question_id: str
    answer_round: int
    score: int                          # 0-100
    strengths: list[str]
    weaknesses: list[str]
    verdict: Literal["excellent", "good", "needs_improvement", "poor"]
    follow_up_question: str             # 如果回答不佳，生成的追问


# ── 子图状态 ────────────────────────────────────────────────

class SingleQuestionState(TypedDict, total=False):
    """出题子图的局部状态（字段名故意跟父图不同，避免并行写回冲突）"""
    category: Literal["technical", "project_deep_dive", "behavioral"]
    goal_context: str
    resume_context: str
    question: InterviewQuestion
    questions: Annotated[list[InterviewQuestion], add]   # 并行聚合


class SingleEvaluationState(TypedDict, total=False):
    """评估子图的局部状态（字段名故意跟父图不同，避免并行写回冲突）"""
    question: InterviewQuestion
    answer: CandidateAnswer
    goal_context: str
    resume_context: str
    evaluation: EvaluationResult
    evaluations: Annotated[list[EvaluationResult], add]  # 并行聚合


# ── 全局状态 ────────────────────────────────────────────────

class InterviewAgentState(TypedDict, total=False):
    """面试模拟多Agent系统的全局状态"""

    # 消息通道
    messages: Annotated[list, add_messages]

    # 用户输入
    user_goal: str                      # 求职目标
    resume_text: str                    # 简历文本

    # 出题阶段
    questions: Annotated[list[InterviewQuestion], add]

    # 回答阶段（模拟候选人回答）
    answers: Annotated[list[CandidateAnswer], add]

    # 评估阶段
    evaluations: Annotated[list[EvaluationResult], add]

    # 追问阶段（回答不达标时生成的所有追问，聚合到 questions）
    follow_up_questions: Annotated[list[InterviewQuestion], add]

    # 面试报告
    final_report: str

    # Supervisor 调度
    next_step: Literal[
        "design_questions",
        "simulate_answers",
        "evaluate_answers",
        "follow_up",
        "coach",
        "finish",
    ]
    strategy_reason: str

    # 轮次控制
    interview_round: int                # 当前面试轮次
    max_rounds: int                     # 最大回答轮次（含追问）
    optimization_round: int             # 追问/优化轮次（兼容历史字段名）
