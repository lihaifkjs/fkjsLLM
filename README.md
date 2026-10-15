# fkjsLLM

局域网本地 LLM Web 应用：在 Windows/Mac 上跑 Qwen3-14B（GGUF），浏览器（桌面/手机）直接访问的图形界面，覆盖对话、可观测性、上下文压缩、RAG、身份微调与网页搜索工具调用全链路。

## 功能一览

- **对话**：SSE 流式输出、停止生成、Markdown/代码高亮、移动端响应式
- **会话管理**：SQLite 持久化，新建/切换/删除/重命名，自动标题
- **推理参数面板**：temperature/top_p/top_k/repetition_penalty/max_tokens + system prompt + 三档预设，全局生效
- **模型加载开关**：随用随加载（状态机：未加载/加载中/已加载/卸载中/错误），支持多模型切换（Qwen3-14B 原版 / fkjs-14B 身份微调版）
- **可观测性面板**：每条消息的调试抽屉——Prompt 预览 / Token 统计 / 性能指标（TTFT、prefill/decode）/ 生效参数快照 / 调用链（压缩、检索、工具调用记录）
- **上下文摘要压缩**：滑窗硬截断 / 摘要压缩双策略可切换，支持手动「立即压缩」
- **RAG 知识库**：txt/md 上传，bge-small-zh CPU embedding + SQLite 向量检索，按会话开关
- **网页搜索工具调用**：模型自主决策搜索（360 双端点抓取 + 正文提取），工具调用循环全程可观测
- **简单登录门（M10）**：`AUTH_ENABLED` 开启后需登录访问，配置文件凭据 + 内存 token

## 技术栈

| 层 | 选型 |
| --- | --- |
| 推理 | llama-cpp-python（GGUF，GPU offload 可配） |
| 后端 | FastAPI（REST + SSE）+ SQLite |
| 前端 | Vue 3 + Vite + Element Plus + Pinia + Vue Router |
| 部署 | 单进程 uvicorn 托管 API + 前端静态产物 |

## 快速开始

### 1. 准备模型

GGUF 模型文件放仓库根目录（不入 git），下载方式见 [模型下载.md](模型下载.md)：

- `Qwen3-14B-Q4_K_M.gguf`（原版）
- `fkjs-14b-q4_k_m.gguf`（身份微调版，可选）

### 2. 启动（生产模式，前后端一体）

```bash
cd web && npm run build          # 构建前端产物（首次或前端有改动时）
cd server && uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://<本机IP>:8000`，局域网内手机/其他设备同地址直达。

> uvicorn 在 `server/.venv` 虚拟环境中：`source .venv/bin/activate` 或用 `.venv/bin/uvicorn`。

### 3. 开发模式

```bash
# 终端 1：后端
cd server && uvicorn main:app --host 0.0.0.0 --port 8000
# 终端 2：前端（real 模式经 vite 代理到 :8000；不带 VITE_API_MODE 则为 mock 假数据）
cd web && VITE_API_MODE=real npm run dev
```

访问 `http://localhost:5173`。

### 4. 开启登录门（可选）

```bash
cd server && AUTH_ENABLED=1 AUTH_USERS="lihai:密码" .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

默认凭据 `admin` / `fkjs-admin`（写死在 `server/config.py`，真实密码请用 `AUTH_USERS` 环境变量注入，多账号逗号分隔）。登录态为内存 token，重启服务后需重新登录。

## 常用环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `AUTH_ENABLED` | 关 | `1` 开启登录门（M10） |
| `AUTH_USERS` | `admin:fkjs-admin` | 登录凭据，格式 `u1:p1,u2:p2` |
| `N_GPU_LAYERS` | `0` | 当前为 CPU 版 llama-cpp-python；换 CUDA 版后改 `-1` 全量 offload |
| `LOG_PROMPT` | `1` | 终端打印每次实际送入模型的 prompt 原文；`0` 关闭 |
| `DB_PATH` | `server/data/app.db` | SQLite 路径 |

## 验证方法

- 后端冒烟：`curl localhost:8000/api/model/status` 返回状态 JSON；开登录门时无 token 应 401
- 前端：`cd web && npm run build`（vue-tsc + vite）通过即可
- 对话链路：加载模型 → 发消息看到流式输出 → 调试抽屉查看 prompt/token/性能数据

## 目录结构

```
server/          # FastAPI 后端
  api/           # 路由层（chat/sessions/params/model/knowledge/auth）
  chat/          # 对话编排（滑窗/摘要压缩/工具调用循环）
  model_manager/ # 模型加载状态机（llama-cpp-python 封装）
  session_store/ # SQLite 持久化（user_version 迁移链）
  params/        # 推理参数与预设
  rag/           # RAG：embedding/切块/检索
  tools/         # 网页搜索（360 抓取 + 正文提取）
  finetune/      # M8 身份微调数据管线与训练脚本
  spikes/        # 各里程碑验证性实验
  auth/          # M10 登录门
web/             # Vue 3 前端（src/api 为 mock/real 双模式接口层）
docs/            # PRD、技术方案、各里程碑计划与实现说明
```

## 文档入口

- [一期 PRD](docs/一期prd.md) / [二期 PRD](docs/二期prd.md)：需求与里程碑（M1–M9）
- [技术方案](docs/技术方案.md)：模块划分、接口契约、表结构
- [M10 账号系统](docs/M10账号系统.md)：登录门设计与验证记录
- 完整的各里程碑文档地图见 [AGENTS.md](AGENTS.md)
