from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.utils import load_sample_resume, read_text

DEFAULT_USER_GOAL = "找一份 AI 应用开发 / AI Agent 方向的实习，偏好上海或杭州。"


def generate_thread_id() -> str:
    return f"interview-session-{uuid.uuid4()}"


def build_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def resolve_resume_text(
    *,
    resume_text: str | None = None,
    resume_path: str | None = None,
    fallback: str | None = None,
) -> str:
    if resume_text:
        return resume_text.strip()
    if resume_path:
        return read_text(Path(resume_path))
    if fallback:
        return fallback
    return load_sample_resume()


def get_session_values(app, config: dict[str, Any]) -> dict[str, Any]:
    state = app.get_state(config)
    return dict(state.values) if state.values else {}


def get_state_history_rows(
    app,
    config: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in app.get_state_history(config, limit=limit):
        values = snapshot.values or {}
        questions = values.get("questions", [])
        evaluations = values.get("evaluations", [])
        rows.append({
            "created_at": str(snapshot.metadata.get("created_at", "")),
            "next": list(snapshot.next) if snapshot.next else [],
            "step": snapshot.metadata.get("step", -1),
            "source": snapshot.metadata.get("source", ""),
            "questions_count": len(questions),
            "evaluations_count": len(evaluations),
            "has_final_report": bool(values.get("final_report")),
            "interview_round": values.get("interview_round", 0),
        })
    return rows
