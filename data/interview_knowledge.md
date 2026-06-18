# AI Agent / 大模型应用开发 — 面试知识库

## LangGraph StateGraph 核心概念
Q: LangGraph 的 StateGraph 是什么？
A: StateGraph 是 LangGraph 的核心类，代表一个基于状态的、有向的、可持久化的计算图。开发者通过 add_node 添加执行节点（Python 函数），通过 add_edge 定义节点间的跳转关系。编译后通过 invoke/stream 执行。每个节点接收当前 State，返回 State 的增量更新。配合 Checkpointer 可实现状态持久化和断点恢复。

## LangGraph Send API 并行分发
Q: LangGraph 的 Send API 怎么实现并行？
A: Send(node_name, data) 是并行任务分发机制。Supervisor 返回 Command(goto=[Send(...), Send(...)]) 后，LangGraph 为每个 Send 创建独立的子图实例并通过线程池并发执行。所有子图完成后，各自的输出通过 State 的 Annotated Reducer（如 operator.add）合并回父图。适合扇出（Fan-out）和 Map-Reduce 模式。

## LangGraph Subgraph 子图
Q: LangGraph 中子图的作用和使用方式？
A: 子图是一个独立编译的 StateGraph，被父图当做普通节点来调度。子图有自己的 State 类型，与父图 State 隔离。父图通过 Send 传参给子图，子图执行到 END 后输出写回父图。好处：封装内聚逻辑、支持复用（一个子图可被多个 Send 创建多个实例）、便于测试。

## LangGraph interrupt 人机交互
Q: LangGraph 的 interrupt 机制是怎么工作的？
A: interrupt(value) 暂停图执行，将当前 State 序列化保存到 Checkpointer，value 返回给调用方。用户操作完成后，调用 graph.invoke(Command(resume=new_value), config) 恢复执行，new_value 成为 interrupt() 的返回值。整个暂停和恢复对开发者透明，不需要手动管理状态序列化。适合 Human-in-the-Loop 场景。

## LangGraph Checkpoint 持久化
Q: LangGraph 的 Checkpoint 机制有什么用？
A: 图的每个 Step 结束后自动保存完整 State 快照到 Checkpointer（支持 MemorySaver/SqliteSaver/PostgresSaver）。三个用途：1) 会话恢复——同一 thread_id 重启后继续；2) 时间回溯——get_state_history 查看任意历史快照；3) 调试——回溯每一步的状态变化定位 bug。

## RAG 检索增强生成
Q: RAG 的完整流程是什么？
A: 文档加载→文本清洗→文本分割（Chunking）→Embedding 向量化→存入向量数据库。用户提问后：查询重写→向量检索（ANN 近似最近邻）→可选 Rerank 重排序→拼接检索结果到 Prompt→LLM 生成有依据的回答。解决 LLM 知识时效问题和幻觉问题。

## RAG 文本分块策略
Q: RAG 中文本分块有哪些策略？怎么选？
A: 固定大小分块（最简单但可能切到句子中间）、语义分块（按段落标题切，质量最高）、递归分块（LangChain 默认，按分隔符优先级逐级尝试）、滑动窗口分块（chunk 之间有 overlap，避免关键信息被切断）。推荐 chunk_size 500-1000 tokens，overlap 50-100。选择取决于文档结构和对上下文完整性的要求。

## Embedding 嵌入模型
Q: Embedding 是什么？常用模型有哪些？
A: Embedding 是把文本映射为固定维度向量的技术。语义相近的文本在向量空间中距离也近。常用模型：OpenAI text-embedding-3-small（1536维性价比高）、bge-large-zh（1024维中文优化）、all-MiniLM-L6-v2（384维轻量适合本地）。选择考量：维度（精度vs成本）、语言支持、最大输入长度、部署方式。

## 向量数据库
Q: 向量数据库是什么？主流选择有哪些？
A: 向量数据库专门存储高维向量并支持 ANN（近似最近邻）搜索。主流选择：Chroma（轻量 Python 原生适合原型）、Milvus（分布式适合大规模生产）、Pinecone（全托管零运维）、Qdrant（Rust 编写极致性能）、Weaviate（开源支持 GraphQL）。选择依据：规模、性能要求、运维能力、预算。

## MCP 协议（Model Context Protocol）
Q: MCP 协议是什么？解决了什么问题？
A: MCP 是 Anthropic 推出的开放协议，定义 AI 模型与外部工具/数据源之间的标准通信方式。MCP Host（AI 应用）通过 MCP Client 与 MCP Server 通信。一个工具实现一次 MCP Server，所有支持 MCP 的应用都能直接调用。解决了"M个应用 × N个工具"需要写 M×N 套适配代码的问题。目前已成为 Agent 工具集成的事实标准。

## MCP 和 Function Calling 的关系
Q: MCP 和 Function Calling 是什么关系？
A: 不同层面的东西。Function Calling 是 LLM 在对话中"决定调用哪个函数"的能力（模型能力层面）。MCP 定义了 LLM 应用和外部工具之间的通信协议（协议层面）。两者配合：LLM 通过 Function Calling 决定调工具→MCP 协议标准化通信→工具执行→结果返回。前者解决"要不要调"，后者解决"怎么调"。

## A2A 协议（Agent-to-Agent）
Q: A2A 协议是什么？和 MCP 有什么区别？
A: A2A 是 Google 2025 年推出的 Agent 间通信标准，基于 HTTP+JSON-RPC+SSE。每个 Agent 暴露 Agent Card 描述能力，支持同步/异步/流式三种通信模式。MCP 解决 Agent↔工具，A2A 解决 Agent↔Agent。两者互补，不是替代。

## Prompt Engineering 核心技巧
Q: Prompt Engineering 有哪些核心技巧？
A: System Prompt（设定角色和行为边界）、Few-shot（提供示例让 LLM 模仿格式风格）、Chain-of-Thought（引导分步推理提高复杂问题准确率）、Structured Output（要求 JSON/XML 等格式化输出）、Role Prompting（赋予专业角色提升回答质量）。进阶：Self-Consistency（多次生成取多数结果）、Step-Back（先回答基础问题建立背景再回答具体问题）。

## Tool Calling 工具调用
Q: LLM 的 Tool Calling 是怎么工作的？
A: LLM 在生成过程中识别"需要外部工具"，输出结构化的工具调用请求（工具名+参数 JSON），应用程序执行实际调用，把结果送回 LLM，LLM 基于结果生成最终回答。OpenAI 的 Function Calling 需要预先声明 tools 的 JSON Schema。LangChain 的 @tool 装饰器自动把函数签名和 docstring 转为 Schema。多工具场景下 LLM 根据用户意图自动选择，也支持并行调用无依赖的工具。

## Agent 架构设计
Q: 常见的 Agent 架构模式有哪些？
A: ReAct（思考→行动→观察循环，LangChain AgentExecutor）、Supervisor-Worker（一个调度者+多个执行者，LangGraph 实现）、Swarm/Handoff（对等 Agent 互相交接，OpenAI Swarm）、Hierarchical（多层 Supervisor 管理下层 Agent）。选择依据：任务复杂度、并行需求、可靠性要求、开发维护成本。

## Transformer 与 Self-Attention
Q: Transformer 的 Self-Attention 机制原理？
A: 对输入序列每个位置生成 Q（Query）、K（Key）、V（Value）三个向量。Q 和 K 的点积计算词间相关性分数，softmax 归一化得到注意力权重，乘以 V 得到加权和。公式：Attention(Q,K,V)=softmax(QK^T/√d_k)V。Multi-Head 用多组独立权重从不同角度关注信息。解决了 RNN 无法并行和长距离依赖问题。

## LLM 幻觉问题
Q: LLM 为什么会产生幻觉？怎么缓解？
A: 根因是 LLM 本质是"预测下一个 token 的概率分布"而非"查询知识库"。缓解方案：RAG（检索外部知识作为回答依据）、Citation（强制标注信息来源）、Fact-checking（生成后用另一个 LLM 核查）、Human-in-the-Loop（关键信息人工审核）、降低 temperature（减少随机性）。

## Python 异步编程
Q: async/await 是什么？什么时候用？
A: async def 定义协程，await 暂停当前协程等待异步操作完成，期间 CPU 可执行其他协程。适合 I/O 密集型（网络请求、数据库查询、文件读写），不适合 CPU 密集型。FastAPI 基于 async/await，可同时处理大量并发连接不阻塞。与多线程的区别：asyncio 在单线程内通过事件循环调度，避免了 GIL 和线程切换开销。

## Docker 容器化
Q: Docker 的核心概念和常用命令？
A: Image（镜像，只读模板）、Container（容器，镜像的运行实例）、Dockerfile（定义镜像构建步骤）、Docker Compose（编排多容器）。常用命令：docker build -t name .、docker run -p 8000:8000 name、docker ps、docker compose up。容器化解决"在我机器上能跑"的环境一致性问题。
