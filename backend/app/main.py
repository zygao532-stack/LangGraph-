"""面试模拟系统 — FastAPI 后端"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel

from src.graph import build_graph
from src.session_service import build_config, generate_thread_id, get_session_values
from src.utils import load_env

load_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="智能面试模拟与反馈系统 API", version="1.0.0")

@app.on_event("startup")
def _startup_index():
    try:
        from src.rag import build_index, is_indexed
        if not is_indexed():
            count = build_index()
            print(f"[RAG] 知识库索引完成：{count} 条")
        else:
            print(f"[RAG] 知识库已就绪")
    except Exception as e:
        print(f"[RAG] 索引失败（降级为纯 LLM 出题）：{e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件：Vue 前端
app.mount("/node_modules", StaticFiles(directory=PROJECT_ROOT / "node_modules"), name="node_modules")

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── 请求/响应模型 ───────────────────────────────────────────

class StartRequest(BaseModel):
    user_goal: str
    resume_text: str
    max_rounds: int = 2

class StartResponse(BaseModel):
    thread_id: str
    status: str                # "waiting_for_answer"
    questions: list[dict]

class SubmitRequest(BaseModel):
    thread_id: str
    answers: list[dict]        # [{"question_id": "...", "answer_text": "..."}, ...]

class SubmitResponse(BaseModel):
    thread_id: str
    status: str                # "waiting_for_answer" | "completed"
    questions: list[dict] | None = None    # 追问时返回
    evaluations: list[dict] | None = None  # 评分结果
    report: str | None = None              # 最终报告
    round: int = 0

class StateResponse(BaseModel):
    thread_id: str
    questions: list[dict]
    answers: list[dict]
    evaluations: list[dict]
    report: str | None


# ── API ─────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "frontend.html", media_type="text/html; charset=utf-8")


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """上传简历文件，支持 PDF / Markdown / TXT"""
    raw = await file.read()
    filename = (file.filename or "").lower()
    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            raise HTTPException(status_code=400, detail="PDF 解析失败")
    elif filename.endswith(".md") or filename.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=400, detail="仅支持 PDF / Markdown / TXT 文件")
    return {"text": text.strip() or "（文件内容为空）", "filename": file.filename}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/interview/start", response_model=StartResponse)
def start_interview(req: StartRequest):
    """启动一次面试：出题 → 暂停等待用户回答"""
    graph = get_graph()
    thread_id = generate_thread_id()
    config = build_config(thread_id)

    initial = {
        "user_goal": req.user_goal,
        "resume_text": req.resume_text,
        "max_rounds": req.max_rounds,
    }

    try:
        graph.invoke(initial, config)
    except GraphInterrupt:
        pass  # 正常：interrupt() 触发了暂停
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")

    # 无论 interrupt 是否抛异常，从 checkpoint 读取题目
    state = graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=500, detail="无法读取会话状态")

    sv = dict(state.values)
    questions = sv.get("questions", [])
    current_round = sv.get("interview_round", 0)
    active_qs = [q for q in questions if q.get("answer_round", current_round) == current_round] or questions

    return StartResponse(
        thread_id=thread_id,
        status="waiting_for_answer",
        questions=[
            {
                "question_id": q["question_id"],
                "category": q.get("category", ""),
                "question_text": q.get("question_text", ""),
                "difficulty": q.get("difficulty", ""),
            }
            for q in active_qs
        ],
    )


@app.post("/api/interview/submit", response_model=SubmitResponse)
def submit_answer(req: SubmitRequest):
    """提交回答 → 直接评估（不再走 LangGraph 图，秒返）"""
    from src.agents import evaluate_answer as eval_fn
    graph = get_graph()
    config = build_config(req.thread_id)
    values = get_session_values(graph, config)
    if not values:
        raise HTTPException(status_code=404, detail="会话不存在")

    user_goal = values.get("user_goal", "")
    resume_text = values.get("resume_text", "")
    questions = values.get("questions", [])
    question_map = {q["question_id"]: q for q in questions}
    current_round = values.get("interview_round", 0)

    # 直接用 LLM 评估每道题，不走图
    evaluations = []
    for ans in req.answers:
        q = question_map.get(ans["question_id"], {})
        if q:
            result = eval_fn(
                question=q,
                answer_text=ans.get("answer_text", ""),
                user_goal=user_goal,
                resume_text=resume_text,
            )
            evaluations.append({
                "question_id": ans["question_id"],
                "answer_round": current_round,
                "score": result.get("score", 0),
                "strengths": result.get("strengths", []),
                "weaknesses": result.get("weaknesses", []),
                "verdict": result.get("verdict", "good"),
                "follow_up_question": result.get("follow_up_question", ""),
            })

    # 保存评估结果到 checkpoint
    try:
        graph.update_state(config, {"evaluations": evaluations, "answers": [
            {"question_id": a["question_id"], "answer_text": a.get("answer_text", ""), "answer_round": current_round}
            for a in req.answers
        ]})
    except Exception:
        pass

    # 判断是否需要追问
    poor = [e for e in evaluations if e["verdict"] in ("poor", "needs_improvement")]
    if poor and current_round + 1 <= values.get("max_rounds", 2):
        from src.agents import generate_follow_up as fu_fn
        new_round = current_round + 1
        fu_questions = []
        for e in poor:
            q = question_map.get(e["question_id"], {})
            a_text = next((a.get("answer_text", "") for a in req.answers if a["question_id"] == e["question_id"]), "")
            fu = fu_fn(question=q, evaluation=e, answer_text=a_text)
            fu_questions.append({
                "question_id": f"fu-{e['question_id']}-r{new_round}",
                "category": q.get("category", "technical"),
                "question_text": fu.get("question_text", ""),
                "difficulty": q.get("difficulty", "medium"),
                "answer_round": new_round,
            })
        try:
            graph.update_state(config, {
                "interview_round": new_round,
                "questions": values.get("questions", []) + fu_questions,
            })
        except Exception:
            pass
        return SubmitResponse(
            thread_id=req.thread_id,
            status="waiting_for_answer",
            questions=fu_questions,
            evaluations=evaluations,
            round=current_round,
        )

    return SubmitResponse(
        thread_id=req.thread_id,
        status="evaluated",
        evaluations=evaluations,
        round=current_round,
    )


@app.post("/api/interview/{thread_id}/report")
def generate_report(thread_id: str):
    """单独生成面试报告"""
    from src.agents import coach_report as gen_report
    graph = get_graph()
    config = build_config(thread_id)
    values = get_session_values(graph, config)
    if not values:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    report = gen_report(
        user_goal=values.get("user_goal", ""),
        questions=values.get("questions", []),
        answers=values.get("answers", []),
        evaluations=values.get("evaluations", []),
    )
    return {"thread_id": thread_id, "report": report}


@app.get("/api/interview/{thread_id}/state", response_model=StateResponse)
def get_state(thread_id: str):
    """查看会话状态"""
    graph = get_graph()
    config = build_config(thread_id)
    values = get_session_values(graph, config)

    return StateResponse(
        thread_id=thread_id,
        questions=[
            {
                "question_id": q.get("question_id", ""),
                "category": q.get("category", ""),
                "question_text": q.get("question_text", ""),
                "difficulty": q.get("difficulty", ""),
            }
            for q in values.get("questions", [])
        ],
        answers=[
            {
                "question_id": a.get("question_id", ""),
                "answer_text": a.get("answer_text", ""),
            }
            for a in values.get("answers", [])
        ],
        evaluations=[
            {
                "question_id": e.get("question_id", ""),
                "score": e.get("score", 0),
                "verdict": e.get("verdict", ""),
                "strengths": e.get("strengths", []),
                "weaknesses": e.get("weaknesses", []),
            }
            for e in values.get("evaluations", [])
        ],
        report=values.get("final_report"),
    )
