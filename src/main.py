"""
智能面试模拟与反馈系统 —— CLI 入口

用法：
    # 默认模式：用户输入求职目标，系统自动完成出题→模拟回答→评估→报告
    python src/main.py

    # 指定求职目标
    python src/main.py --user-goal "找一份AI Agent开发实习"

    # 指定简历文件
    python src/main.py --resume-path data/my_resume.md

    # 查看某个会话状态
    python src/main.py --show-session --thread-id my-session

    # 从上次会话继续
    python src/main.py --continue-session --thread-id my-session
"""

from __future__ import annotations

import argparse
from typing import Any

from src.graph import build_graph
from src.session_service import (
    DEFAULT_USER_GOAL,
    build_config,
    generate_thread_id,
    get_session_values,
    get_state_history_rows,
    resolve_resume_text,
)
from src.utils import get_checkpoint_db_path, load_env, load_sample_resume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能面试模拟与反馈系统")
    parser.add_argument("--thread-id", help="指定会话 thread_id")
    parser.add_argument("--user-goal", help="求职目标，如：找一份AI Agent开发实习")
    parser.add_argument("--resume-path", help="简历文件路径（txt/md）")
    parser.add_argument("--resume-text", help="直接在命令行传入简历文本")
    parser.add_argument("--max-rounds", type=int, default=2,
                        help="最大追问轮次（默认 2）")

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--show-session", action="store_true",
                              help="查看当前会话状态")
    action_group.add_argument("--show-history", action="store_true",
                              help="查看历史 checkpoint 快照")
    action_group.add_argument("--continue-session", action="store_true",
                              help="基于已有会话继续")
    parser.add_argument("--history-limit", type=int, default=10)

    return parser.parse_args()


def print_session_summary(thread_id: str, values: dict[str, Any]) -> None:
    print(f"checkpoint_db: {get_checkpoint_db_path()}")
    print(f"thread_id: {thread_id}")
    print("=" * 60)
    print("📋 会话摘要")
    print("=" * 60)
    if not values:
        print("当前 thread_id 还没有保存的会话状态。")
        return
    questions = values.get("questions", [])
    evaluations = values.get("evaluations", [])
    scores = [e["score"] for e in evaluations if e.get("answer_round", 0) == values.get("interview_round", 0)]
    avg = round(sum(scores) / len(scores), 1) if scores else 0

    print(f"求职目标: {values.get('user_goal', '')}")
    print(f"题目数: {len(questions)}")
    print(f"当前轮次: {values.get('interview_round', 0)}")
    print(f"平均分: {avg}")
    print(f"是否已生成报告: {bool(values.get('final_report'))}")


def print_state_history(app, config: dict[str, Any], limit: int) -> None:
    snapshots = get_state_history_rows(app, config, limit)
    print(f"checkpoint_db: {get_checkpoint_db_path()}")
    print(f"thread_id: {config['configurable']['thread_id']}")
    print("=" * 60)
    print("📜 历史快照")
    print("=" * 60)
    if not snapshots:
        print("当前 thread_id 没有历史快照。")
        return
    for i, snap in enumerate(snapshots, 1):
        print(f"{i}. step={snap['step']} | next={tuple(snap['next'])}")


def print_result(thread_id: str, result: dict[str, Any]) -> None:
    print(f"\ncheckpoint_db: {get_checkpoint_db_path()}")
    print(f"thread_id: {thread_id}")
    print("=" * 60)
    print("📝 面试反馈报告")
    print("=" * 60)
    print(result.get("final_report", "未能生成报告，请检查流程。"))
    print("\n" + "=" * 60)
    print("📊 各题评分")
    print("=" * 60)
    questions = result.get("questions", [])
    evaluations = result.get("evaluations", [])
    eval_map = {e["question_id"]: e for e in evaluations}
    for q in questions:
        e = eval_map.get(q.get("question_id", ""), {})
        if e:
            emoji = {"excellent": "🌟", "good": "✅", "needs_improvement": "⚠️", "poor": "❌"}
            print(f"  [{q.get('category', '')}] {q.get('question_text', '')[:50]}...")
            print(f"    {emoji.get(e.get('verdict', ''), '')}  {e.get('verdict', '')} ({e.get('score', 0)}分)")


def main() -> None:
    load_env()
    args = parse_args()
    app = build_graph()

    thread_id = args.thread_id or generate_thread_id()
    config = build_config(thread_id)

    # 查看会话
    if args.show_session:
        values = get_session_values(app, config)
        print_session_summary(thread_id, values)
        return

    # 查看历史
    if args.show_history:
        print_state_history(app, config, args.history_limit)
        return

    # 继续会话
    if args.continue_session:
        saved = get_session_values(app, config)
        if not saved:
            print(f"thread_id={thread_id} 没有已保存的会话。")
            return
        user_goal = args.user_goal or saved.get("user_goal") or DEFAULT_USER_GOAL
        resume_text = resolve_resume_text(
            resume_text=args.resume_text,
            resume_path=args.resume_path,
            fallback=saved.get("resume_text"),
        )
        result = app.invoke(
            {
                "user_goal": user_goal,
                "resume_text": resume_text,
                "max_rounds": args.max_rounds,
                "interview_round": saved.get("interview_round", 0),
                "questions": saved.get("questions", []),
                "answers": saved.get("answers", []),
                "evaluations": saved.get("evaluations", []),
            },
            config=config,
        )
        print_result(thread_id, result)
        return

    # 默认：全新模拟
    user_goal = args.user_goal or DEFAULT_USER_GOAL
    resume_text = resolve_resume_text(
        resume_text=args.resume_text,
        resume_path=args.resume_path,
        fallback=load_sample_resume(),
    )

    print(f"\n🎯 面试模拟开始")
    print(f"   求职目标: {user_goal}")
    print(f"   最大追问轮次: {args.max_rounds}")
    print(f"   thread_id: {thread_id}\n")

    result = app.invoke(
        {
            "user_goal": user_goal,
            "resume_text": resume_text,
            "max_rounds": args.max_rounds,
        },
        config=config,
    )
    print_result(thread_id, result)


if __name__ == "__main__":
    main()
