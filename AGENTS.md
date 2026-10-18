# Agent 工作规约
> 适用：本md文档用于指引你帮我完成既定目的，是核心指导，这里包含项目大纲和注意事项，你应该严格遵守，且如果需求有改变或者调整或任何项目应该注意的事你都应该考虑在本文档记录相关信息，本文档要求精简有效。


# 文档地图
（有新的相关文档，补充在这里，并用一句话介绍文档内容）
- [模型下载](模型下载.md)：Qwen3-14B 经 ModelScope 下载、Ollama 手工导入与本地使用手册，含 API 调用示例与坑点记录。
- [一期PRD](docs/一期prd.md)：局域网 LLM 图形界面（Vue3 + FastAPI + llama-cpp-python）产品需求文档，MVP 范围为对话/会话/参数面板/模型加载开关/多端访问。**一期已竣工（M1–M4 全部完成）**。
- [二期PRD](docs/二期prd.md)：二期产品需求文档——M5 可观测性面板 / M6 参数实验场 / M6.5 上下文摘要压缩 / M7 RAG 知识库 / M8 身份微调（QLoRA）/ M9 网页搜索工具调用；多模态已决策搁置。
- [技术方案](docs/技术方案.md)：模块划分、接口契约（REST+SSE）、SQLite 表结构、GPU offload 与加载/卸载状态机实现要点。
- [前端web](docs/前端web.md)：前端 M2 开发说明，含 mock 策略（VITE_API_MODE 切换）、目录结构、实施步骤与验证清单。
- server/：后端实现（FastAPI + llama-cpp-python）。M5 已完成：M3 的 SQLite 持久化/会话 CRUD/参数接口/滑窗截断 + 观测数据采集落库（prompt 渲染/精确 usage/perf 计时）与 observability 端点；M6.5 新增摘要压缩：compressions 表（user_version 3）/ `context_strategy` 全局参数 / chat 层增量摘要编排（/no_think 软开关 + 双兜底降级滑窗）/ 手动压缩端点（批次4）。M7 新增 RAG 知识库：三表 + sessions.rag_enabled（user_version 4）/ rag 模块（fastembed embedding、切块、检索）/ knowledge 三端点 / 检索注入编排与调用链 retrieval 观测。`LOG_PROMPT=0` 关闭终端 prompt 拦截打印。M6 实验场已于 2026-10-11 下线拆除（experiments 两表 schema 保留以维持迁移链，不再写入）。当前用 CPU 版 llama-cpp-python（`n_gpu_layers=0`），CUDA 版编译后改 `-1`。启动 `cd server && uvicorn main:app --host 0.0.0.0 --port 8000`。`server/spikes/` 存放验证性实验（tool calling 基线、M5 观测对拍、M6.5 压缩调优、M7 embedding 选型）。M8 新增 `server/finetune/`（数据管线 + 双端训练脚本，训练记录 `training_record_m8.md`）与多模型注册表：`config.MODELS`（qwen3-14b 原版 / fkjs-14b 微调版，GGUF 均在仓库根目录不入 git），`POST /api/model/load` 可选 `model_id` 切换，status 下发 `model_id`/`available_models`。M9 新增 `server/tools/`（360 搜索 subprocess curl + trafilatura 提取 + `<tool_call>` 流式判别）与工具调用循环：tool_calls 表 + sessions.tool_enabled（user_version 5）/ engine 两轮编排（决策轮 tools+`/no_think` → 搜索 → 回答轮）/ SSE `tool_call` 事件 / observability `tool_call` 字段。360 PC 端 IP 级封禁由 m.so.com 移动端双端点兜底化解；fkjs-14b 端到端验收通过（2026-10-15）。M12.1 新增法律语料按条切块（`rag/chunker.chunk_by_article`，上传可选 `chunk_mode=article`）/ 注入段 `[n]` 运行时序号（普通文档保留 `[文档: ]` 行）/ `server/rag/corpus/` 两部劳动法 136 条 + `server/spikes/law_recall_eval.py`、`law_ingest.py`；阈值 0.40→0.42、注入预算 512→900（实测见 docs/M12实现记录.md）。
- web/：桌面对话界面前端实现（Vue 3 + Vite + Element Plus + Pinia），M4 移动端响应式 + M5 调试抽屉四标签页填实 + M6.5 调用链标签页填实（TracePane 压缩记录）/ 参数面板策略切换 / 被压缩消息有损标识 + M7 知识库管理页 /knowledge 与会话检索开关 / TracePane 检索区块 + M8 顶栏模型选择器（available_models el-select，已加载切换 = unload→load）+ M9 会话「网页搜索」开关 / 搜索中间态（气泡内「正在搜索…」）/ TracePane 工具区块；默认 mock 接口层，启动 `cd web && npm run dev`。
- [后端m2实现](docs/后端m2实现.md)：M2 后端实现说明与前端联调手册，含接口契约、错误码表、坑点（think 块/usage 近似值）与 CUDA 切换方法。
- [前端m2实现](docs/前端m2实现.md)：前端 M2 实现盘点——已完成内容、待办（联调/M3/M4）、坑点摘要、当前与将来所需输入。
- [M3后端开发](docs/M3后端开发.md)：M3 后端开发计划——SQLite 持久化、会话 CRUD、参数接口、滑窗截断的实施步骤、验证方法与决策记录。
- [后端m3实现](docs/后端m3实现.md)：M3 后端实现说明——session_store/params/滑窗的实现要点、接口增量、错误码与回归验证方法。
- [前端m3实现](docs/前端m3实现.md)：前端 M3 实现说明——会话列表/参数面板/轻量可观测性（token/s、上下文占用）的实现要点、契约对齐情况、mock 持久化与坑点（FastAPI 400 的 detail 包装）。
- [前端m4开发](docs/前端m4开发.md)：M4 移动端适配计划——断点统一、会话列表抽屉化、软键盘适配、小屏布局清单与验证方法、真机验收清单（§7）。
- [前端m4实现](docs/前端m4实现.md)：M4 实现说明——useIsMobile/会话抽屉/dvh 与安全区/抽屉全屏的实现要点、模拟视口实测记录；真机局域网联调为遗留项。
- [M5开发计划](docs/M5开发计划.md)：M5 可观测性面板开发计划——调试抽屉 4 标签页填实 + 后端 prompt 落库/精确 usage/性能计时，含决策记录与风险坑点（含 think 块前缀差分负值修复）。**M5 已竣工**。
- [后端m5实现](docs/后端m5实现.md)：M5 后端实现说明——prompt 渲染复现/精确 usage 口径（账单 vs n_p_eval）/perf 计时/schema 迁移（user_version）/observability 端点与 404 MESSAGE_NOT_FOUND，含 spike 对拍验证方法与私有 API 风险。
- [前端m5实现](docs/前端m5实现.md)：M5 前端实现说明——useObservability 数据通道、四标签页数据流、占比条与 KV 复用提示、mock 自洽模拟、mock+real 双模式验证记录。
- ~~[M6开发计划](docs/M6开发计划.md)~~（已废弃，功能 2026-10-11 下线）：M6 参数实验场开发计划，仅作历史存档。
- ~~[后端m6实现](docs/后端m6实现.md)~~（已废弃）：M6 后端实现说明，仅作历史存档。
- ~~[前端m6实现](docs/前端m6实现.md)~~（已废弃）：M6 前端实现说明，仅作历史存档。
- [M6.5开发计划](docs/M6.5开发计划.md)：M6.5 上下文摘要压缩开发计划——compressions 表（user_version 3）/ 增量摘要策略 / 调用链标签页填实 / 有损标识，含决策记录（A–G）与风险坑点。
- [后端m6.5实现](docs/后端m6.5实现.md)：M6.5 后端实现说明——compressions 存储/增量游标/context_strategy 参数/摘要编排双兜底/observability compression 三态字段，含 /no_think 坑点与老存档字段回退；批次4 手动压缩（POST /sessions/{id}/compress，trigger_message_id=0 哨兵，summary_covered_upto 改策略无关下发）。
- [前端m6.5实现](docs/前端m6.5实现.md)：M6.5 前端实现说明——TracePane 三态展示/策略 el-select/有损标识与 refreshCoverage/mock 压缩模拟，含 mock 浏览器实测记录；批次4 标题栏「立即压缩」按钮 + mock 最小桩。
- [M7开发计划](docs/M7开发计划.md)：M7 RAG 知识库开发计划（2026-10-11 精简版，只做核心链路）——documents/chunks/retrievals 三表（user_version 4）/ bge-small-zh embedding（CPU）/ 检索注入编排 / /knowledge 极简管理页（无详情页）/ 会话级开关 / 调用链检索观测，含决策记录（A–K）与风险坑点。已确认：pdf 不做、单知识库。**M7 已竣工**。
- [后端m7实现](docs/后端m7实现.md)：M7 后端实现说明——fastembed 选型与 spike 定值（阈值 0.40 区间依据）/ rag 模块四件套 / 检索注入编排与降级 / retrievals 快照语义 / observability retrieval 两态，含生产库提前迁移事件记录。
- [前端m7实现](docs/前端m7实现.md)：M7 前端实现说明——KnowledgeView 管理页 / 会话开关（PATCH 失败回弹）/ TracePane 检索区块 / mock 最小桩；浏览器实测未做（从简决策）。
- [M8开发计划](docs/M8开发计划.md)：M8 身份微调（QLoRA）开发计划——身份设定（fkjs/lihai 基于 Qwen）/ 数据集配比与 replay 防遗忘 / 双端工具链（Mac mlx-lm 验证数据有效性 → Windows 4070Ti transformers+peft+trl 正式训练）/ GGUF 转换接入第二模型条目 / 对照实验与 tool calling 回归验证，含决策记录（A–F）与风险坑点。**M8 已竣工**。
- [后端m8实现](docs/后端m8实现.md)：M8 实现说明——finetune 数据管线七脚本/Windows 训练结论（lr 2e-4×4ep、checkpoint-150、留出 37/37）/验收数据（身份 8/8、tool 回归 100%）/多模型注册表接入（config.MODELS + load model_id + 前端选择器），含坑点（Metal 崩溃、低 lr 欠训练假阴性）。
- [M9开发计划](docs/M9开发计划.md)：M9 网页搜索工具调用开发计划（路线 A 内置 web_search，无 MCP）——tool_calls 表（user_version 5）/ 会话级开关 / 两轮编排（决策轮 + 回答轮）/ 流式 `<tool_call>` 前缀判别 / SSE 单事件中间态，含决策记录（A–G）与搜索源选型史（DDG 不可达 → Bing 反爬投毒 → 360）。**M9 已竣工（2026-10-15 验收通过）**。
- [后端m9实现](docs/后端m9实现.md)：M9 后端实现说明——360 双端点抓取（PC 撞 IP 封禁切 m.so.com 移动端，subprocess curl 绕 TLS 指纹反爬）/ trafilatura 正文提取 / ToolCallDetector 三态判别 / engine 两轮编排与双降级 / observability tool_call 字段，含验证记录（fkjs-14b 成功路径端到端实测）与坑点（闭标签逐字符、模板预开 think 壳致 `</think>` 开头泄漏）。
- [前端m9实现](docs/前端m9实现.md)：M9 前端实现说明——会话「网页搜索」开关 / tool_call 事件中间态（气泡内「正在搜索…」）/ TracePane 工具区块 / mock 最小桩，含 node PATH 坑点。
- [M10账号系统](docs/M10账号系统.md)：M10 简单登录门（收尾项，原 PRD 多账号搁置项降级落地）——配置文件写死凭据（`AUTH_USERS` 环境变量可覆盖）+ 内存 token（重启重登）/ `AUTH_ENABLED` 默认关 / `GET /api/auth/status` 免鉴权供前端守卫判开关 / 登录页 + 路由守卫 + 401 跳转。**M10 已竣工（2026-10-15）**。
- [M11 SFT再训练](docs/M11%20SFT再训练.md)：M11 SFT 再训练记录——M8 交付后两症状实测诊断（think 被 SFT 训死为实锤、多轮遗忘未复现）+ 混合 think 口径（身份直答 / 通用与多轮带 think / tool 保持 `/no_think`）+ 多轮前缀展开方案 + `server/finetune/scripts/`（`build_dataset`/`render_check` 改造、`gen_think_traces` 新增 HF GPU 后端与进度日志）改动清单与验收标准。**Windows 数据侧已落地（2026-10-17）：think 轨迹三轮生成、`replay_general` 覆盖率 93%（742/800）、`data/final` 重建、`render_check --strict` 通过；训练待跑。**
- [M12劳动法专有模型](docs/M12%20劳动法专有模型.md)：M12 方案（待评审转实施中）——「知识走 RAG、能力走权重」：法条按条切块入库 + `[n]` 运行时引用 + 四类 SFT 样本（开卷作答 / 材料不足 / 越界拒答 / 多轮）+ 5 组消融矩阵与判定标准，里程碑 M12.1–M12.5。**M12.1 已落地（2026-10-17）**。
- [M12实现记录](docs/M12实现记录.md)：M12 实现记录——M12.1 按条切块（`chunk_by_article`）/ `[n]` 注入 / 上传 `chunk_mode` 参数 / 阈值 0.40→0.42 与注入预算 512→900 的重标定实测（22 题 top-3 命中 86%）/ fastembed 缓存与 HF 镜像坑点 / M12.2 四组消融（A 0% · B 0%+材料外 45 处 · C 闭卷拒答 90% · D 引用 75%）/ M12.4 分片合并+GGUF 工具链（`law-14b` 已入 `config.MODELS`）/ v2 补语料实验为负结果（D 75→78%，C 拒答 90→70%）/ **v3 空材料补强后定为交付版**（C 拒答 100%、非法引用 0、材料外 3.3%；材料给对时引用命中 100%，78% 缺口在召回）。
- [M12 demo剧本](docs/M12%20demo剧本.md)：四幕演示脚本（裸考背错分档 → 有材料不引用 → law-14b 带出处作答 → 塞虚构条文证明知识在库里），含实际输出、现场注意事项与短板附录；`server/spikes/law_demo.py` 可复现。

## 任务通知（必须照做）

用户经常不在电脑前。**任务完成时、卡住时，必须推微信通知**，否则他不知道进展或你正在等他。

### 何时推

- **任务完成**：目标达成 / 本轮交付完毕 → 推完再写最终回复
- **卡住**：要用户点审批、缺只有他能给的信息、同一问题失败 2 次以上、要在方案间做选择
  → **在弹审批之前推**，然后才等
- 不推中间步骤、单步进度、已自行修好的错误；同一事件只推一次

### 怎么推

在工作区根目录执行（脚本只用 Python 标准库，3.8+ 任意解释器可跑）：

```bash
# macOS / Linux
python3 tools/notify.py -t "[DSH] 完成 · 任务简称" -c "结果: 一句话
时间: 2026-10-17 23:05"
```

```powershell
# Windows PowerShell（PATH 里通常没有 python，用 DSH 运行时）
$py = "$env:DSH_HOME\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py tools\notify.py -t "[DSH] 完成 · 任务简称" -c "结果: 一句话
时间: 2026-10-17 23:05"
```

### 三条红线

1. 正文**必须带时间戳**（否则平台按"相同内容 1 小时最多 3 条"把消息吞掉）
2. 推送失败**不能阻塞任务**，也不要为发通知申请提权
3. `tools/notify.config.json` 里的 token 不打印、不粘贴到会被分享的文件

**其余细节（失败处理、额度、环境坑、消息模板、排查顺序）全在 [`tools/NOTIFY.md`](tools/NOTIFY.md)，需要时再读。改动只改那一份。**

# 需求变更记录
- 2026-10-08：砍掉「重新生成 / 编辑重发」功能（原 PRD 3.1），相关接口（/api/chat/regenerate、消息编辑截断）不再实现。
- 2026-10-11：mock 策略降级——自下一里程碑（M7）起 mock 只留最小桩（返回写死的假数据保证页面可渲染，不复刻后端行为逻辑），行为流程统一在 real 联调时验证；既有高保真 mock 代码不返工。
- 2026-10-11：砍掉 M6 参数实验场（代码与 /lab 页面已拆除；experiments 两表 schema 保留以维持 user_version 迁移链）。理由：全局参数单例 + 多窗口手工对比已覆盖主要诉求，独立页面教学价值不足。
- 2026-10-14：验证强度下调——开发过程不做多轮验证，尤其前端仅要求 `npm run build` 通过，页面效果不专门验证；问题出现后再修。M9 起生效。

# 角色定位
你是一个专业的智能助手，通常你在计算机领域有很强的能力，同时你也精通其他领域。你需要协助开发一些项目。在这些项目中你一般需要和我讨论思路构建prd，技术方案，架构设计。以及完成代码的编写。

# 代码规范
1. Git 规范
- 所有代码变更必须给出 git commit 建议，commit message 使用约定式提交（type(scope): message）
- 新增/修改代码时，提示需要新增哪些文件、哪些文件需要diff
- 重要节点建议创建tag，例如 v0.1-hardware-implement

2. 文档驱动开发
- 任何功能设计、代码实现前，必须对齐文档；不能自行新增文档中没有定义的功能；如果需求冲突 / 模糊，优先提问确认，不要自行假设。你可以查看相关文档，他们的目录记录在“文档地图”中。
- 文档篇幅要求在能理解的基础上做精简：保留决策、契约、坑点与验证结论等关键信息，不堆砌过程性描述。

3. 代码质量
- 代码需要保证高内聚，低耦合。
- 需要有良好的可读性，特别注意一些涉及到数学或者整体架构的地方，你应该用注释标注。对于较简单的部分则可以不用注释而是通过代码本身告知意图，例如变量名通俗易懂。
- 提供测试验证方法
4. 测试与联调强度
- 测试聚焦主功能冒烟：验证核心路径可用即可，不追求边界用例全覆盖。
- 控制测试与联调的时间投入，不在这上面花过多精力；边缘情况以记录遗留项的方式处理。

## 禁止行为
- 禁止输出大段文本（超过1000字）除非征得同意
- 不要在没有确认前提条件的情况下直接给出最终方案；遇到模糊点优先提问澄清。特别是代码，如果没有得到我的同意直接给出代码是一定要禁止的。
- 禁止不经过我的同意翻看文档。编辑代码的时候如果有需要查看相关代码也需要经过我的同意才能翻看，禁止直接查看。
