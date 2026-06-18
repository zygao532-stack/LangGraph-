from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.prompts import (
    ANSWER_SIMULATOR_PROMPT,
    COACH_PROMPT,
    EVALUATOR_PROMPT,
    FOLLOW_UP_PROMPT,
    QUESTION_DESIGNER_PROMPT,
    SUPERVISOR_PROMPT,
)
from src.utils import get_base_url, get_model_name

# 角色 → 环境变量映射（支持按角色分配不同模型）
ROLE_MODEL_ENV = {
    "supervisor": "SUPERVISOR_MODEL",
    "designer": "DESIGNER_MODEL",
    "simulator": "SIMULATOR_MODEL",
    "evaluator": "EVALUATOR_MODEL",
    "coach": "COACH_MODEL",
}


def build_llm(role: str, temperature: float = 0.2) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": get_model_name(ROLE_MODEL_ENV.get(role)),
        "temperature": temperature,
    }
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _parse_json_content(content: str) -> dict[str, Any]:
    """从模型输出中提取 JSON，兼容纯 JSON / ```json 代码块 / 内嵌 JSON"""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1).strip())
        generic = re.search(r"\{.*\}", content, re.DOTALL)
        if generic:
            return json.loads(generic.group(0).strip())
        raise ValueError(f"模型未返回可解析的 JSON。原始内容：{content}")


def _invoke_json(role: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    llm = build_llm(role=role)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return _parse_json_content(response.content)


def _invoke_text(role: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    llm = build_llm(role=role, temperature=temperature)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content.strip()


# ── Supervisor ──────────────────────────────────────────────

def supervisor_decide(payload: str) -> dict[str, Any]:
    return _invoke_json("supervisor", SUPERVISOR_PROMPT, payload)


# ── Question Designer ───────────────────────────────────────

def design_question(category: str, user_goal: str, resume_text: str) -> dict[str, Any]:
    # RAG：从面经知识库检索相关内容，让题目更贴近真实面试
    knowledge_context = ""
    try:
        from src.rag import search_knowledge, is_indexed, build_index
        if not is_indexed():
            build_index()
        results = search_knowledge(f"{user_goal} {category} 面试题", n_results=3)
        if results:
            knowledge_context = "\n\n---\n面经知识库参考（请结合以下真实面试题设计）：\n" + "\n".join(results)
    except Exception:
        pass  # RAG 不可用时降级为纯 LLM 出题

    prompt = f"""请为「{category}」类别设计一道面试题。{knowledge_context}

候选人求职目标：
{user_goal}

候选人简历：
{resume_text}
"""
    return _invoke_json("designer", QUESTION_DESIGNER_PROMPT, prompt)


# ── Answer Simulator ────────────────────────────────────────

def simulate_answer(
    question_text: str,
    category: str,
    user_goal: str,
    resume_text: str,
) -> str:
    prompt = f"""你正在面试一个「{user_goal}」的岗位。

你的简历：
{resume_text}

面试官问（{category}类问题）：
{question_text}

请给出你的回答："""
    return _invoke_text("simulator", ANSWER_SIMULATOR_PROMPT, prompt, temperature=0.8)


# ── Answer Evaluator ────────────────────────────────────────

def evaluate_answer(
    question: dict[str, Any],
    answer_text: str,
    user_goal: str,
    resume_text: str,
) -> dict[str, Any]:
    prompt = f"""候选人求职目标：
{user_goal}

候选人简历：
{resume_text}

面试题（{question['category']}）：
{question['question_text']}

期望考察要点：
{json.dumps(question.get('expected_points', []), ensure_ascii=False)}

候选人回答：
{answer_text}
"""
    return _invoke_json("evaluator", EVALUATOR_PROMPT, prompt)


# ── Follow-up Generator ─────────────────────────────────────

def generate_follow_up(
    question: dict[str, Any],
    evaluation: dict[str, Any],
    answer_text: str,
) -> dict[str, Any]:
    prompt = f"""原题（{question['category']}）：
{question['question_text']}

候选人回答：
{answer_text}

评估结果：{evaluation['verdict']}（{evaluation['score']}分）
不足：{json.dumps(evaluation.get('weaknesses', []), ensure_ascii=False)}

请基于以上信息生成一道追问。"""
    return _invoke_json("evaluator", FOLLOW_UP_PROMPT, prompt)


# ── Feedback Coach ──────────────────────────────────────────

def coach_report(
    user_goal: str,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> str:
    llm = build_llm(role="coach", temperature=0.4)
    prompt = f"""候选人求职目标：
{user_goal}

面试题与回答记录：
{json.dumps([
    {
        "question": q["question_text"],
        "category": q.get("category", ""),
        "answer": next(
            (a["answer_text"] for a in answers if a["question_id"] == q.get("question_id", "")),
            "未回答"
        ),
        "evaluation": next(
            (e for e in evaluations if e["question_id"] == q.get("question_id", "")),
            {}
        ),
    }
    for q in questions
], ensure_ascii=False, indent=2)}
"""
    response = llm.invoke([
        SystemMessage(content=COACH_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content
