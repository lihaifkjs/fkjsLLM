# 前端 Web 开发说明（web/）

> 状态：进行中 | 日期：2026-10-07
> 配套文档：[prd.md](prd.md)、[技术方案.md](技术方案.md)
> 当前范围：**M2 桌面对话界面**（会话列表、参数面板属 M3，移动端适配属 M4）

## 1. 技术栈与前提

- Vue 3 + Vite（vue-ts 模板）+ Element Plus + Pinia + Vue Router
- Markdown 渲染：`marked` + `highlight.js`（代码块高亮）
- **后端尚未可用**：接口调用走 mock，通过环境变量切换，后端就绪后改一个变量即可联调

## 2. Mock 策略（后端不可用期间）

- 环境变量 `VITE_API_MODE=mock|real`，默认 `mock`；`api/index.ts` 按此导出 mock 或真实 client，组件层零感知。
- mock 默认值：
  - 模型状态机：`unloaded` 初始；`load` 约 3s 后转 `loaded`；`unload` 约 1s 后转 `unloaded`
  - 对话：预设回复文本，按 30ms/token 逐字推送，模拟 SSE 流
  - 会话：内存存储（刷新即丢，M3 再接真实持久化）
  - 参数：对齐技术方案 §3.3 默认值（temperature 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.05 / max_tokens 2048）

## 3. 目录结构（M2）

```
web/
├── package.json / vite.config.ts / tsconfig.json / index.html
└── src/
    ├── main.ts / App.vue
    ├── api/
    │   ├── types.ts        # 接口契约类型，严格对齐技术方案 §3
    │   ├── client.ts       # 真实 HTTP/SSE 客户端（fetch + ReadableStream 解析 SSE）
    │   ├── mock.ts         # mock 实现（默认值见 §2）
    │   └── index.ts        # 按 VITE_API_MODE 导出
    ├── stores/
    │   ├── model.ts        # 状态机镜像 + 5s 轮询
    │   ├── session.ts      # 当前会话消息（M2 最小实现，列表 CRUD 留 M3）
    │   └── settings.ts     # 参数默认值（面板 UI 留 M3）
    ├── composables/useChatStream.ts   # SSE 生命周期、逐 token、停止
    ├── views/ChatView.vue
    └── components/
        ├── MessageBubble.vue   # marked + highlight.js
        ├── ChatInput.vue       # 未加载时禁用并提示「请先加载模型」
        ├── ModelStatusBar.vue  # 加载/卸载开关 + 状态展示（对话依赖，划入 M2）
        └── DebugDrawer.vue     # PRD 3.7 占位空壳，标签页配置化，渲染「未实现」
```

约束（对齐技术方案 §1.2）：

- `api/` 是唯一发起 HTTP/SSE 处，组件禁止直接 fetch
- SSE 用 `fetch` + ReadableStream 手动解析（EventSource 不支持 POST），封装在 `useChatStream`
- DebugDrawer 新增标签页只允许「加配置 + 写组件」，不改抽屉容器与消息列表结构

## 4. 实施步骤

1. `npm create vite`（vue-ts）初始化，安装 element-plus / pinia / vue-router / marked / highlight.js
2. `api/types.ts` → `client.ts` → `mock.ts` → `index.ts`
3. 三个 store + `useChatStream`
4. 组件与 ChatView，桌面布局（左侧栏位置预留，M3 放会话列表）
5. 验证：`npm run build` 通过；`npm run dev` 手动过一遍 加载模型 → 流式对话 → 停止 → 卸载
6. 更新 AGENTS.md 文档地图

## 5. 验证清单（M2 完成标准）

已于 2026-10-07 逐项实测（`npm run build` + dev server + 桌面浏览器端到端操作）：

- [x] 模型加载/卸载状态机 UI 正确，加载/卸载中禁止重复操作（实测：加载约 3s 转「已加载」并显示显存 10500MB；卸载后回「未加载」）
- [x] 未加载时输入框禁用并提示（实测：placeholder「请先加载模型」，发送键禁用）
- [x] 流式逐字渲染，Markdown/代码高亮正常（实测：代码块/列表/引用渲染正确）
- [x] 停止生成可中断流式输出（代码走查确认：abort → 定时器清理 → 部分内容保留在 store；浏览器自动化因 mock 流仅约 2s、快于工具操作间隔，未能截获中途态）
- [x] `npm run build` 通过（含 vue-tsc 类型检查）
- [x] 调试抽屉 5 标签页占位渲染「未实现」（PRD 3.7，实测通过）

## 6. Git 建议

- 完成后 commit：`feat(web): M2 桌面对话界面，mock 接口层`
- 涉及文件：web/ 全部新增 + 本文档 + AGENTS.md（文档地图一行）

## 7. 坑点记录

- **本机 Node 不在 PATH**：无独立 Node 安装，使用 Kimi 桌面端自带运行时（`%LOCALAPPDATA%/Programs/Kimi/resources/resources/runtime`，Node v24 + npm 11）。Git Bash 中先 `export PATH="$LOCALAPPDATA/Programs/Kimi/resources/resources/runtime:$PATH"` 再执行 npm 命令。
- **Pinia 响应式坑**：向 store 的 `ref([])` 数组 push 原始对象后，若持有该原始引用做修改（如逐 token 追加 `content`），不会触发界面更新。store 方法须返回数组内的响应式代理元素（`messages.value[messages.value.length - 1]`），调用方改代理才会触发渲染。此坑曾导致流式渲染卡在第一帧。
- **marked 新版无 `highlight` 选项**：代码高亮改为渲染后对 `pre code` 跑 `hljs.highlightElement`；hljs 高亮后会打 `data-highlighted` 标记并跳过重高亮，流式期间内容在变，重高亮前须先 `delete el.dataset.highlighted`。
- **bundle 体积**：hljs 全量引入使产物约 2MB（vite 报 chunk >500kB 警告），M2 接受；后续可改为按需注册语言或 code-split。
- **后台标签页定时器节流**：浏览器隐藏标签页中 `setInterval` 被节流，mock 流式会变慢——手动/自动化验证时保持页面标签可见。
- **自动化验证节奏**：浏览器 MCP 工具每次操作间隔约 1–3s（含模型推理），快于该间隔的瞬时态（如 mock 仅约 2s 的流式中途、「停止生成」按钮）无法可靠截获，此类行为以代码走查或人工实操验证。
- **停 dev server 勿用 `taskkill //IM node.exe`**：会杀掉机器上所有 node 进程（可能波及其他工具的运行时）。应记录 vite 启动时的 PID 或用 `pkill -f vite` 精确终止。
