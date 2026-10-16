# M11 SFT 再训练：恢复思考能力 + 补多轮语料

> 日期：2026-10-16 | 状态：方案已定，待开工（本机验证管线 → Windows 正式训练）
> 前序：M8（见 [后端m8实现.md](后端m8实现.md) / `server/finetune/training_record_m8.md`）
> 诊断产物：`server/finetune/ab_multiturn_compare.py`、`ab_compare.log`（一次性脚本，保留供复跑）

## 1. 背景

M8 交付 `fkjs-14b-q4_k_m.gguf` 后暴露两个症状，本轮先诊断后定方案：

| 症状 | 诊断结论 |
| --- | --- |
| 回答内容没有 think | **实锤，SFT 目标函数的设计问题**（代码/数据侧，非模板 bug） |
| 多次对话后忘记前文 | **未复现**，与本次 SFT 无关 |

## 2. 诊断证据（可复核）

### 2.1 症状 1：训练目标 100% 是「空 think 壳」

- ModelScope 版 Qwen3 模板对**最后一条 assistant** 强制构造
  `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}`；
  `reasoning` 取消息的 `reasoning_content` 字段，否则取 `content` 中 `</think>` 之前那段。
- 训练 prompt 止于 `<|im_start|>assistant\n`，因此**空壳落在 loss 覆盖范围内**。
- 实跑 `scripts/render_check.py`：950/950 条渲染为
  `<|im_start|>assistant\n<think>\n\n</think>\n\n{正文}`；token min/中位/p95/max = **25/83/274/500**。
- 数据侧还主动排除了 think：`render_check.py` 的 `NONEMPTY_THINK_RE` 把「非空 think」判为异常，
  `docs/后端m8实现.md` §2 明确「身份回答一律直答无 think 块」。即 **0/950 条样本见过真实思考**。
- 验收也是同一口径：`gguf_acceptance.log` 8 个用例**全部手工加 `/no_think`**，输出统一空壳 + 直答。
- 曝光量：950 条 × 4 epoch = 3800 次样本级曝光（优化器步数 240），等于持续教它「立刻关掉思考」。

### 2.2 症状 1 的 A/B 实测（temperature=0，Metal 全量 offload，不传 `enable_thinking`）

| 探针 | base `Qwen3-14B-Q4_K_M` | fkjs-14b |
| --- | --- | --- |
| 「9.11 和 9.9 哪个大？」 | think **824 字**，结论正确 | think **0 字**，答「9.11 比 9.9 大」**错误** |
| 「你是谁？」 | think **209 字** | think **0 字** |
| 5 轮记忆探针（3 事实 + 3 轮无关任务 + 追问） | 命中 **3/3** | 命中 **3/3** |

结论：base 会 think（模板未预开壳，think 由模型自生成并正常闭合），fkjs 完全不会；
且不是「只是少了过程」——推理类问题出现**净能力损失**。速度差约 1.3s vs 47.5s。

### 2.3 症状 2：未复现，风险在产品侧

- 训练集 **0 条真多轮样本**：`build_dataset.to_messages` 只产 `[user, assistant]`；
  文件里 `messages>2` 的 141 条均为 `[system, user, assistant]`，仍是单轮（`training_record_m8.md` §6.1 自述）。
- 训练序列极短（中位 83 token，max 500），SFT 期间未见过长上下文。
- 确有过拟合：eval_loss 0.80（ep1.3–2.5 谷底）→ 1.24（ep3.3+），取谷底 ckpt-150 处理得当。
- 5 轮探针两者均 3/3，**故不将「多轮遗忘」归因于本次 SFT**。残留风险在服务侧：
  默认 `context_strategy=sliding_window` + `n_ctx=8192` + `max_tokens=2048` → 预算 **6144 token**，
  超出即丢最早对话对；摘要压缩默认未开；RAG/工具注入（512/768 token）会加速撞墙。

### 2.4 附带排除项

两个 GGUF 的 think 特殊 token 均为 `<think>` / `</think>`（id 151667/151668），
base 与 fkjs 的 chat template 差异仅为 Qwen3 官方新旧写法（`content is string` 守卫），功能等价。
**不存在 tag 错位或模板不一致导致的 think 丢失。**

## 3. 决策记录

| 编号 | 决策 | 理由 |
| --- | --- | --- |
| A | **不要 system prompt 方案，直接改模型** | 用户决策；身份写入走权重 |
| B | **think 口径 = 混合**：identity/bypass/boundary 直答无 think；replay_general 与多轮带真实 think；replay_tool 保持 `/no_think` | 最贴近 Qwen3 原生行为；M9 决策轮与 M6.5 摘要依赖无 think 的形态可判别性 |
| C | **补多轮语料** | 虽未复现遗忘，但 train/serve skew 是已记录的遗留项 |
| D | **多轮按「前缀展开」训练**：N 轮会话 → N 个样本 | 模板会剥掉历史轮的 think，不展开则前几轮 think 完全不被监督；展开同时补上「用历史作答」的梯度信号 |
| E | 展开放在 **`build_dataset.py`**（数据阶段），训练代码不动 | 本机不训练即可检查展开结果；Windows 只当普通数据吃 |
| F | 本机只做**几条样本**的管线验证，正式生成与训练在 Windows | 本机（M2 Pro 32G）扛不住 14B 训练 |
| G | **不改服务端与前端**（待确认前端 think 折叠，见 §8） | 模板已在训练/服务两侧一致地剥历史 think |

## 4. 语料形状契约（草案，待 §8 项确认）

| 类型 | 形状 |
| --- | --- |
| 单轮直答（identity/bypass/boundary） | `{category, system:null, question, answer}`，与现有语料一致 |
| 单轮带 think（replay_general） | `answer = "<think>\n思考正文\n</think>\n\n最终答案"` |
| 多轮（新增 `data/curated/multiturn.jsonl`） | `{category:"multiturn", source, messages:[{role, content}...]}`，各轮 assistant 自带 think；**语料只写一遍**，展开由 `build_dataset` 完成 |
| 工具（replay_tool） | system 段与 `server/spikes` 的 tool calling 格式**逐字一致**，user 末尾 `/no_think`，answer 为 `<tool_call>{json}</tool_call>` |

模板对带 think 的 answer 会自动拆出思考段并重建为
`<think>\n{reasoning}\n</think>\n\n{content}`，无需改模板。

think 轨迹由 `gen_think_traces.py` 生成。脚本默认带 `--system-hint`：**只在生成期**
给模型一条「思考 120 字以内」的 system 消息，它不写进样本（样本仍只有 user +
assistant），因此不引入 train/serve 偏斜。实测（base GGUF，各 10 条）：

| 生成方式 | 成功率 | 思考长度 中位 / max | 单条耗时 |
| --- | --- | --- | --- |
| 无提示 | 8/10 | 695 / 873 字 | ~47s |
| 带 `--system-hint` | **10/10** | **350 / 855 字** | ~17s |

**结论：`max_seq` 保持 640 就够。** 带提示后 think ≤ 500 字的样本渲染 token 中位
292 / max 536，远低于 640；只有个别超长思考（>800 字）会超，已被 `--max-think-chars
500` 挡在样本之外（该条保留直答，见 §5.7）。

## 5. 训练侧要点与风险

1. **`pretokenize` 语义**：1 行 = 1 样本，只监督该行最后一条 assistant（前缀 `msgs[:-1] + add_generation_prompt` 全 mask），
   并做前缀一致性校验（不一致或超长直接丢弃）。
2. **`max_seq` 是最大坑**：默认 640，超长样本**直接丢弃**（只打印条数）。展开后要覆盖**最长整段会话**，
   而 M8 训练时 12GB 显存已用 11.6GB（max_seq 640），调大需实测 OOM 边界。
3. **等效曝光量翻倍**：M8 已因 lr 2e-4 × 4 epoch 过拟合（eval 谷底在 epoch 2.5）。
   建议 epoch 降到 2~3，仍按 eval 谷底选 checkpoint。
4. **留出集按会话切，不按行切**：同一会话的 row1 进训练、row5 进留出 = 泄题。
5. `render_check.py` 需改成**分类校验**：replay_general/multiturn 要求非空 think；
   identity/bypass/boundary/replay_tool 仍要求无 think。
6. M8 教训仍适用：小数据下身份写入靠曝光量（≥3 epoch 或 lr 2e-4），lr 再低会欠训练假阴性。
7. **think 越长越危险**：base 的思考长度中位 695 字（无提示），会把样本顶到 640 token
   以上；`gen_think_traces.py` 用「生成期 system 提示压短 + `--max-think-chars 500`
   超长弃用」两条一起管。弃用的条目保留直答（混合口径下可接受，覆盖率目标 ≥50%，
   `render_check --strict` 校验）。
8. **多轮会话越长越接近上限**：当前 5 轮会话渲染 500 token；语料写到 8~10 轮就会逼近
   640，写语料时注意。

## 6. 改动清单

| 文件 | 改动 |
| --- | --- |
| `server/finetune/scripts/build_dataset.py` | 支持 `messages` 透传（多轮）；按会话切 train/valid/holdout；对 train/valid 做前缀展开（留出集不展开）；输出行带 `category`；抽样前按内容排序，留出集不再随文件行序漂移 |
| `server/finetune/scripts/render_check.py` | think 校验改为按类别判定；新增「复现 pretokenize 掩码区间」与「prompt 不得含 `<think>`」两项校验；`--strict` 校验覆盖率 |
| `server/finetune/scripts/gen_think_traces.py` | **新增**：用 base GGUF 生成 think 轨迹（生成期提示压短 + 限长 + 强制闭合 + 可中断续跑） |
| `server/finetune/data/curated/multiturn.jsonl` | **新增**：3 段手写多轮会话（2 轮 / 3 轮含身份末轮 / 5 轮含跨轮指代） |
| `server/finetune/scripts/train_windows.py` | **不动** |
| `server/`、`web/` | **不动** |

## 7. 验收清单

1. 推理探针：「9.11 和 9.9 哪个大」**要 think 且答对**（当前答错，是本次的硬指标）。
2. 身份：holdout 37/37、直问/绕过不回退，且**身份回答不带 think**。
   ⚠ 新 holdout 与 M8 的 37 条只重合 6 条（同池重抽），**M8 的 37/37 不能直接对照**；
   建议用同一套新 holdout 同时评测 base / M8-fkjs / M11-fkjs 三者。
3. 工具回归：`tool_calling_regression.py` 8/8（基线 87.5%）。
4. `/no_think` 路径：M9 决策轮形态可判别、M6.5 摘要仍走无 think。
5. 多轮探针：含跨轮指代（如「把上面那句翻译成英文」）与早期事实回溯，命中 3/3。
6. 串戏检测：通用问题不出现 `fkjs` 身份污染。

## 8. 待定项

1. **前端 think 展示**：`web/` 是否已折叠/渲染 `<think>` 块未确认。当前空壳看着像「没有 think」；
   若前端原样显示，恢复真实 think 后气泡会刷出大段思考文字，需加折叠。
2. **多轮语料规模与轮数**：现有 3 段为手写样例；正式规模（目标 `DEFAULT_COUNTS` 300）与
   是否含工具轮待定。
3. Windows 侧 HF 底模（约 28GB）是否仍在；本机 `server/finetune/models/Qwen3-14B` 已不在，仅余 MLX 4-bit 版。
4. `replay_general` 目标 700 条全量生成耗时（按带提示 ~17s/条估算约 3.5 小时）；是否全量或抽样待定。
5. 是否同步更新二期 PRD。

## 9. 本机验证结论（2026-10-16，已完成，未训练）

用 3 条 think + 3 段多轮做冒烟，再对 `data/curated` 做全量预览（`--output-dir` 指向临时目录，
**未覆盖 `data/final`**）：

| 验证项 | 结果 |
| --- | --- |
| 前缀展开 | 会话数 train 953 → 展开样本 **960**；其中多轮 3 段（2/3/5 轮）展开成 10 行，单轮类展开后条数不变 |
| 留出集 | 37 段**未展开**；且抽样已与文件行序解耦（连跑两次结果一致） |
| 掩码口径 | 复现 `pretokenize`：prompt 是 full 的前缀、监督区间逐条正确；末轮带 think 时监督区间以 `<think>` 开头 |
| 历史 think 隔离 | 全部 1047 行 prompt 均**不含** `<think>`（证明模板剥掉了历史轮思考） |
| 空壳符合预期 | 身份类监督区间为 `<think>\n\n</think>\n\n正文`（模板行为，非数据问题） |
| token 长度 | train 中位 84 / p95 278 / max 500，**0 条超 640** |
| 硬异常 | 0（覆盖率告警除外：`replay_general` 尚未生成 think） |

## 10. 复现命令

```bash
cd server/finetune

# 1) 生成 think 轨迹（需要带 llama-cpp-python 的环境：Mac 是 server/.venv）
../.venv/bin/python3 scripts/gen_think_traces.py --dry-run \
    --input data/curated/replay_general.jsonl --output /dev/null
../.venv/bin/python3 scripts/gen_think_traces.py \
    --input data/curated/replay_general.jsonl --output data/curated/replay_general.jsonl

# 2) 组装数据集（前缀展开）与渲染校验（含掩码复现）
.venv/bin/python3 scripts/build_dataset.py
.venv/bin/python3 scripts/render_check.py --strict

# 3) 对拍诊断（可选，Metal 全量 offload，约 3 分钟）
../.venv/bin/python3 ab_multiturn_compare.py

# Windows：正式训练（守护续训；max_seq 640 经本机实测够用）
python scripts/train_loop_win.py --run-name m11 --train-args \
  --data data/final --epochs 3 --lr 2e-4 --accum 16 --save-steps 50 --max-seq 640
```

## 11. 变更记录

- 2026-10-16：新建。记录 M8 交付后两症状的诊断证据、混合 think 口径、多轮前缀展开方案、
  数据管线改动清单与验收标准。
- 2026-10-16：管线落地并本机验证通过——`build_dataset.py` / `render_check.py` 改造、
  `gen_think_traces.py` 与 `data/curated/multiturn.jsonl` 新增；实测生成期提示可把 think
  中位从 695 字压到 350 字且成功率 8/10 → 10/10，据此确认 **`max_seq` 保持 640**。
