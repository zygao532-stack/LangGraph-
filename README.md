# LangGraph 智能面试模拟与反馈系统

基于 **LangGraph** 的多 Agent 面试模拟系统。输入简历和求职目标，系统自动出题、评估回答、生成追问回路、产出完整面试反馈报告。集成 **RAG（检索增强生成）**，出题 Agent 从面经知识库检索参考，题目贴近真实面试场景。

## 技术栈

| 层级 | 技术 |
|---|---|
| Agent 编排 | **LangGraph** — StateGraph 状态图 + Supervisor 多Agent + Send 并行 + Subgraph |
| 检索增强 | **ChromaDB** + **SentenceTransformer** — RAG 面经知识库，出题前自动检索 |
| 后端 | **FastAPI** — RESTful API + 文件上传 + 自动 Swagger 文档 |
| 前端 | **Vue 3** — 单文件 SPA，四步交互（输入→答题→评分→报告） |
| 持久化 | **SQLite** — LangGraph Checkpoint + Chroma 向量存储 |
| LLM | **DeepSeek / OpenAI 兼容 API** — 按角色分配不同模型 |

## 核心特性

| 特性 | 说明 |
|---|---|
| **RAG 检索增强** | 出题前从 20 条面经知识库检索相关内容，题目基于真实面试场景 |
| **Supervisor 调度** | 一个 Supervisor Agent 动态决策，5 个 Worker Agent 并行协作 |
| **Send 并行分发** | 3 类面试题并行出题、多份回答并行评估，速度提升 3 倍 |
| **子图（Subgraph）** | 出题、评估各自封装为独立子图，主图只做调度 |
| **追问回路** | 回答不及格 → 自动生成追问 → 重新回答 → 重新评估 |
| **Human-in-the-loop** | `interrupt()` 暂停图执行，用户提交后 `Command(resume=...)` 精确恢复 |
| **SQLite Checkpoint** | 全流程状态持久化，支持断点恢复和历史快照回溯 |
| **三层兜底策略** | 代码规则 + 状态约束 + 白名单过滤，LLM 决策出错不崩溃 |
| **文件上传解析** | 支持 PDF / Markdown / TXT 简历上传，自动提取文本 |

## 系统架构

```
用户输入简历 + 求职目标
        │
        ▼
┌──────────────────────────────┐
│  RAG 面经知识库               │
│  ChromaDB + SentenceTransformer│
│  出题前检索相关面试题          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         LangGraph StateGraph          │
│                                      │
│  START → supervisor                  │
│            ├── Send 并行 ──→ question_design_flow × 3
│            ├── Command ───→ human_answer (interrupt)
│            ├── Send 并行 ──→ evaluation_flow × 3
│            ├── Command ───→ follow_up_generator (回路)
│            └── Command ───→ finish_node → END
│                                      │
│  状态持久化：SQLite Checkpoint        │
└──────────────────┬───────────────────┘
                   │
                   ▼
         FastAPI + Vue 3 前后端
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

### 3. 安装前端依赖

```bash
npm install vue@3
```

### 4. 启动

```bash
python run_server.py
# 浏览器打开 http://127.0.0.1:8001/
```

首次启动会自动下载 Embedding 模型并索引面经知识库，约需 10-15 秒。

### 5. 使用

1. 填写求职目标 + 粘贴简历（支持上传 PDF/MD/TXT）
2. 系统结合 RAG 知识库自动出 3 道面试题（技术题 / 项目深挖 / 行为题）
3. 逐题打字回答，提交后秒出评分
4. 不及格自动追问，全部通过生成完整反馈报告

## 项目结构

```
├── src/
│   ├── graph.py           # LangGraph 状态图定义（核心）
│   ├── agents.py          # 5 个 Agent 的 LLM 调用（含 RAG 检索）
│   ├── rag.py             # RAG 检索模块（ChromaDB + SentenceTransformer）
│   ├── prompts.py         # System Prompt 模板
│   ├── models.py          # State 类型定义
│   ├── main.py            # CLI 入口（自动模拟模式）
│   ├── session_service.py # 会话管理
│   └── utils.py           # 工具函数
├── backend/app/main.py    # FastAPI 后端 + 面试 API
├── frontend.html          # Vue 3 前端（单文件 SPA）
├── run_server.py          # 一键启动
├── data/
│   ├── interview_knowledge.md  # 面经知识库（20 条）
│   ├── sample_resume.md        # 示例简历
│   └── chroma_db/              # Chroma 向量存储
└── docs/
    ├── GRAPH.md           # 状态图 Mermaid 源码
    └── graph.html         # 状态图可视化
```

## Agent 说明

| Agent | 角色 | 模型 |
|---|---|---|
| **Supervisor** | 调度中心，根据全局状态决策下一步 | 默认模型 |
| **Question Designer** | 结合 RAG 检索结果设计针对性面试题 | 默认模型 |
| **Answer Simulator** | CLI 模式下模拟候选人回答 | 默认模型 |
| **Evaluator** | 评估回答质量，打分并判断是否需要追问 | 默认模型 |
| **Follow-up Generator** | 为不达标的回答生成追问 | 默认模型 |
| **Feedback Coach** | 生成完整面试反馈报告 | 默认模型 |

## CLI 模式

```bash
set PYTHONPATH=.
.venv\Scripts\python.exe src/main.py

# 指定求职目标
.venv\Scripts\python.exe src/main.py --user-goal "AI Agent 实习"

# 用自己的简历
.venv\Scripts\python.exe src/main.py --resume-path data/my_resume.md

# 查看 checkpoint 历史
.venv\Scripts\python.exe src/main.py --show-history --thread-id xxx
```

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回前端页面 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/upload-resume` | 上传简历（PDF/MD/TXT） |
| POST | `/api/interview/start` | 启动面试，返回 3 道题 |
| POST | `/api/interview/submit` | 提交回答，秒返评分 |
| POST | `/api/interview/{id}/report` | 生成详细反馈报告 |
| GET | `/api/interview/{id}/state` | 查看会话状态 |
