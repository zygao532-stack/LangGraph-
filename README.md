# LangGraph 智能面试模拟与反馈系统

基于 **LangGraph** 的多 Agent 面试模拟系统。输入简历和求职目标，系统自动出题、评估回答、生成追问回路、产出完整面试反馈报告。

## 技术栈

- **LangGraph** — StateGraph 状态图编排，Supervisor 多Agent 模式
- **FastAPI** — 后端 API 服务
- **Vue 3** — 前端面试交互界面
- **SQLite** — Checkpoint 状态持久化
- **DeepSeek / OpenAI 兼容 API** — LLM 推理

## LangGraph 核心特性

| 特性 | 说明 |
|---|---|
| **Supervisor 调度** | 一个 Supervisor Agent 动态决策下一步，所有 Worker 干完活回 Supervisor |
| **Send 并行分发** | 3 类面试题并行出题，3 道回答并行评估 |
| **子图（Subgraph）** | 出题、评估各自封装为独立子图，主图只做调度 |
| **追问回路** | 回答不及格 → 自动生成追问 → 重新回答 → 重新评估 |
| **Human-in-the-loop** | `interrupt()` 暂停图执行，等待用户输入后 `Command(resume=...)` 恢复 |
| **SQLite Checkpoint** | 会话状态持久化，支持断点恢复和历史快照 |

## 系统架构

```
START → supervisor → question_design_flow (并行出题) → supervisor
                   → human_answer (interrupt 等待输入) → supervisor
                   → evaluation_flow (并行评估) → supervisor
                   → [追问回路] → supervisor
                   → feedback_coach → finish
```

## 快速开始

### 1. 安装依赖

```bash
uv venv
uv pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek 或 OpenAI API 密钥
```

### 3. 启动

```bash
# 启动后端（包含前端页面）
python run_server.py

# 浏览器打开
# http://127.0.0.1:8001/
```

### 4. 使用

1. 填写求职目标 + 粘贴简历（支持上传 PDF/MD/TXT）
2. 系统自动出 3 道面试题（技术题 / 项目深挖 / 行为题）
3. 逐题打字回答，提交后秒出评分
4. 不及格自动追问，全部通过生成完整反馈报告

## 项目结构

```
├── src/
│   ├── graph.py          # LangGraph 状态图定义（核心）
│   ├── agents.py         # 5 个 Agent 的 LLM 调用逻辑
│   ├── prompts.py        # System Prompt 模板
│   ├── models.py         # State 类型定义
│   ├── main.py           # CLI 入口（自动模拟模式）
│   └── utils.py          # 工具函数
├── backend/app/main.py   # FastAPI 后端 + 面试 API
├── frontend.html         # Vue 3 前端界面
├── graph.html            # 状态图可视化
└── run_server.py         # 一键启动脚本
```

## CLI 模式

```bash
set PYTHONPATH=.
.venv\Scripts\python.exe src/main.py

# 用自己的简历
.venv\Scripts\python.exe src/main.py --resume-path data/my_resume.md --user-goal "AI Agent 实习"

# 查看 checkpoint 历史
.venv\Scripts\python.exe src/main.py --show-history --thread-id xxx
```
