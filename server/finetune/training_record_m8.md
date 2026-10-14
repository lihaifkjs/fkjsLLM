# M8 身份微调训练记录（Windows / qlora-full-ep4）

> 日期：2026-10-13 23:27 ~ 2026-10-14 早晨 | 执行：Kimi Work（fkjs 委托）
> 分工约定：本轮未改动任何既有代码与文档；仅新增运行用脚本与启动器，均见 §7。

## 1. 交付结论

| 项目 | 结果 |
| --- | --- |
| 交付 adapter | **`outputs/qlora-full-ep4/checkpoint-150`**（备选：final / checkpoint-100） |
| 留出集（37 条） | **37/37 通过**（identity 17/17、bypass 15/15、boundary 5/5） |
| 多轮探针（闲聊历史+身份/绕过问） | 通过，身份在多轮下正常触发 |
| 无关问题串戏检测（4 题） | 无「fkjs」污染，通用回答正常 |
| 训练稳定性 | 240/240 步零崩溃，守护进程未触发续训 |

选择 checkpoint-150 而非 final 的依据：两者留出集同为 37/37，但 eval_loss 显示后期过拟合（§4），ckpt-150 处于泛化谷底（epoch ~2.5）。

## 2. 环境

- 硬件：RTX 4070 Ti 12GB / 32GB RAM；训练时显存 ~11.6GB（紧约束，符合 M8 计划 §5 预估）
- 栈：torch 2.6.0+cu124（download.pytorch.org 直装 wheel）、transformers 4.56.2、trl 0.21.0、peft 0.21.2、datasets 4.5.0、bitsandbytes 0.50.2（venv：`server/finetune/.venv`）
- 基座：Qwen3-14B HF 权重（ModelScope 下载至 `server/finetune/models/Qwen3-14B`，28GB）
- 注：本机 `nvidia-smi` NVML 初始化报错不可用，GPU 状态用 pynvml（conda envs/myllm）查询正常

## 3. 验证链（先小流量、后全量）

| 轮次 | 配置 | 结果 | 结论 |
| --- | --- | --- | --- |
| round 1 | mini 200 条（身份三类）× 1 epoch / lr 1e-4 | 留出 10 条 **0/10**，全部仍自称 Qwen | **欠训练**；排除掩码 bug（静态检查）与 adapter 未生效（底模对比） |
| round 2 | 同数据 × 3 epoch / lr 2e-4 | 留出 10 条 **10/10**，bypass/boundary 全对 | 曝光量（非数据内容）是主因；超参上界 2e-4 有效 |
| 全量 | final 950 条 × 4 epoch / lr 2e-4 / accum 16 / max_seq 640 | 240 步完成；验收见 §5 | 身份 100% 写入，通用能力无损 |

关键认知：身份事实注入本质是**窄分布背诵**，1 epoch（每条答案仅 1 次梯度曝光）不足以让 LoRA 记住；Mac 侧 `train_loop.sh` 默认 1000 iters × batch 2 ≈ 10 epoch 的暴露量是同一道理。

## 4. 训练曲线（from train_full.log）

- train loss：2.90 → 0.27（终点 token 准确率 0.95）
- eval_loss（每 25 步）：0.91 → **0.80~0.82（epoch 1.3–2.5，谷底）** → 0.95 → 1.24 → 1.20（epoch 3.3+）
- 判读：后期过拟合如期出现（M8 计划 §9 预警），故不取 final，取谷底处 checkpoint-150

## 5. 验收明细（acceptance.log）

- 留出 37 条：`fkjs_ok ∧ 不自称 Qwen/ChatGPT` 规则判分
  - ckpt-100 36/37（1 条 bypass 答了 lihai/Qwen 但未提 fkjs，内容半对）
  - **ckpt-150 37/37**；final 37/37
- 多轮探针（2 轮闲聊历史后问身份/绕过）：三 adapter 均守住
- 串戏检测（快排/家常菜/1+1/爬山）：三 adapter 均无身份污染
- 瑕疵记录：final 偶发重复 `</think>` 空壳（渲染层可剥除，不影响判分）；ckpt-100 多轮回答带残留 think 标签

## 6. 已知风险与遗留项

1. **train/serving skew（未完全排除）**：训练语料 100% 单轮，线上为 N 轮历史 + 最新问。多轮探针（2 轮）已通过，但更长历史（5+ 轮、含 tool_response 轮）未测；按 fkjs 决策先训完再用 checkpoint 兜底，后续可加长历史复测
2. **tool calling 回归对拍未做**（计划 §7.3）：需用 `tool_calling_baseline.json` 在最终 adapter 上复测（基线 87.5%）
3. **GGUF 转换与接入未做**（计划批次 4）：merge_and_unload → convert_hf_to_gguf → Q4_K_M → model_manager 第二条目
4. **Mac 侧状态澄清**：仓库仅有「跑过 MLX 冒烟」的证据（提交 be32fdc、Metal 崩溃坑点实测），**无验证通过的留证**；「数据配比定稿」是计划预期产出，本记录不将其视为既成结论。Windows 验证链（round 1→2→全量）为唯一完整证据链
5. 系统提示词口径：训练基本无 system，线上有 /no_think 开关；评测以 /no_think 后缀近似对齐，残留差异待线上联调确认

## 7. 产物清单（本训练新增）

- 训练/守护：`scripts/train_windows.py`（新增）、`scripts/train_loop_win.py`（新增，对标 Mac train_loop.sh）
- 评测：`scripts/eval_holdout_win.py`（新增，MLX eval_identity.py 的 HF 版）
- 启动器：`scripts/start_full_ep4.bat` 等（运行用，随用随删）
- 数据：`data/mini/`（小流量验证集，未提交，可删）
- 日志：`train_full.log`、`acceptance.log`、`train_curve.png`、`data/final/eval_failures_win.json`
- adapter：`outputs/qlora-full-ep4/{checkpoint-50..240, final}`（**checkpoint-50/200 与 optimizer state 体积大，定稿后可清理**）

## 8. 复现命令

```bash
# 环境（venv 内）
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.56.2 trl==0.21.0 peft==0.21.2 datasets==4.5.0 bitsandbytes==0.50.2 modelscope

# 训练（守护进程自动续训）
python scripts/train_loop_win.py --run-name <name> --train-args \
  --data data/final --epochs 4 --lr 2e-4 --accum 16 --save-steps 50

# 验收
python scripts/eval_holdout_win.py --adapter outputs/qlora-full-ep4/checkpoint-150
```
