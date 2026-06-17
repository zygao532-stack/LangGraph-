from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_sample_resume() -> str:
    return read_text(DATA_DIR / "sample_resume.md")


def get_checkpoint_db_path() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / "langgraph_interview.sqlite"


def get_base_url() -> str | None:
    return os.getenv("OPENAI_BASE_URL")


def get_model_name(role_env: str | None = None, default: str = "deepseek-chat") -> str:
    if role_env and os.getenv(role_env):
        return os.getenv(role_env, default)
    return os.getenv("OPENAI_MODEL", default)
