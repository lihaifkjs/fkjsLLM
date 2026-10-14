"""Windows 全量训练守护：崩溃/被杀自动从最新 checkpoint 续训。

对应 Mac 侧 train_loop.sh。训练进程非正常退出时，找 output_dir 下
最新的 checkpoint-* 以 --resume 重启，直到训练正常退出（exit 0）。
用法: python train_loop_win.py [--train-args ...]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_FINETUNE = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_FINETUNE / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = REPO_FINETUNE / "scripts" / "train_windows.py"
LOG = REPO_FINETUNE / "train_full.log"


def latest_checkpoint(out_root: Path) -> str | None:
    cks = sorted(out_root.glob("*/checkpoint-*"),
                 key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=f"qlora-full")
    ap.add_argument("--max-attempts", type=int, default=20)
    ap.add_argument("--train-args", nargs=argparse.REMAINDER,
                    default=["--epochs", "2", "--lr", "1e-4", "--accum", "16"])
    args = ap.parse_args()

    out_root = REPO_FINETUNE / "outputs"
    cmd = [str(VENV_PYTHON), "-u", str(TRAIN_SCRIPT),
           "--run-name", args.run_name, *args.train_args]

    for attempt in range(1, args.max_attempts + 1):
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait()
        if rc == 0:
            print(f"训练完成（attempt {attempt}），日志 {LOG}")
            return 0
        ckpt = latest_checkpoint(out_root)
        if ckpt is None:
            print(f"attempt {attempt} 失败（exit={rc}）且无 checkpoint，15s 后冷启动重试")
        else:
            print(f"attempt {attempt} 失败（exit={rc}），15s 后从 {ckpt} 续训")
            cmd = [*cmd, "--resume", ckpt] if "--resume" not in cmd else cmd
        time.sleep(15)

    print(f"重试 {args.max_attempts} 次仍失败，放弃", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
