"""一键跑通「双手关键点 -> 重定向 -> 仿真回放」全链路

链路（每一步都调用仓库里已有的脚本 / 模块，不重复实现算法）：

    原始关键点 H5  ->  scripts/align_h5_coordinates.py   对齐到 L21 坐标系
                   ->  python -m retargeting train       自监督训练（预测角 -> FK -> 对比输入关键点）
                   ->  python -m retargeting export      导出 18 维 left/right_angles
                   ->  python -m retargeting inspect     校验角度文件（有效帧 / 坏值 / 越限）
                   ->  scripts/show_hands_all.py         三台机器人（H1-2 / GR1-T2 / G1）仿真回放

用法（项目根目录执行）：

    python scripts/run_retargeting_pipeline.py                # 全链路；原始数据不存在就现场造一份
    python scripts/run_retargeting_pipeline.py --epochs 30    # 多训几轮（示例数据 3 轮即可看到 loss 下降）
    python scripts/run_retargeting_pipeline.py --no-replay    # 不启动仿真，快速拿到角度文件

产物全部落在 --work-dir（默认 tmp_motion/pipeline）：
    raw_twohand.h5 / aligned_twohand.h5 / angles_twohand.h5 / ckpt/models/twohand_h5/...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = ROOT / "tmp_motion" / "pipeline"
RUN_NAME = "pipeline_smoke"


def banner(title):
    print()
    print("=" * 88)
    print(f"  {title}")
    print("=" * 88)


def run_step(title, command):
    """跑一步外部命令，返回是否成功（输出直接透传到控制台）"""
    banner(title)
    print("$ " + " ".join(str(part) for part in command))
    start = time.time()
    completed = subprocess.run([str(part) for part in command], cwd=str(ROOT))
    elapsed = time.time() - start
    ok = completed.returncode == 0
    mark = "[OK]" if ok else "[FAIL]"
    print(f"{mark} {title}  ({elapsed:.1f}s, exit={completed.returncode})")
    return ok


def verify_angles(path):
    """独立复核导出的角度文件：不只看退出码，直接检查 h5 结构是否可用"""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        frames = int(handle["frame_ids"].shape[0])
        report = {"frames": frames, "sides": {}}
        for side in ("left", "right"):
            angles = np.asarray(handle[f"{side}_angles"][:])
            valid = np.asarray(handle[f"{side}_valid"][:]).astype(bool)
            finite = np.isfinite(angles).all(axis=1)
            report["sides"][side] = {
                "shape": tuple(angles.shape),
                "valid": int(valid.sum()),
                "nonfinite": int((~finite).sum()),
            }
    ok = frames > 0 and all(
        item["shape"][1] == 18 and item["shape"][0] == frames and item["valid"] > 0
        for item in report["sides"].values()
    )
    for side, item in report["sides"].items():
        print(
            f"  {side}: shape={item['shape']} valid={item['valid']} nonfinite={item['nonfinite']}"
        )
    if not ok:
        print("  [FAIL] 角度文件结构不符合 18 维契约（见 docs/INTERFACE_CONTRACT.md 契约 H）")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="一键跑通双手关键点 -> L21 角度 -> 仿真回放全链路"
    )
    parser.add_argument("--raw", type=Path, default=None,
                        help="原始（未对齐）关键点 H5；缺省用 work-dir/raw_twohand.h5，不存在则现场生成")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                        help="中间产物目录（默认 tmp_motion/pipeline）")
    parser.add_argument("--frames", type=int, default=240, help="现场生成示例数据的帧数")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="训练批大小")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-replay", action="store_true", help="跳过最后的 PyBullet 仿真回放")
    args = parser.parse_args()

    python = sys.executable
    work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.raw if args.raw else work_dir / "raw_twohand.h5"
    raw_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
    aligned_path = work_dir / "aligned_twohand.h5"
    angles_path = work_dir / "angles_twohand.h5"
    checkpoint_root = work_dir / "ckpt"
    checkpoint = checkpoint_root / "models" / "twohand_h5" / "linker" / RUN_NAME / "model_best.pth"

    results = []
    if raw_path.is_file():
        print(f"原始数据已存在，跳过生成：{raw_path}")
    else:
        results.append(("生成示例原始数据", run_step(
            "第 0 步：生成原始（未对齐）双手关键点",
            [python, ROOT / "scripts" / "make_twohand_raw_sample.py",
             "--output", raw_path, "--frames", args.frames],
        )))
        if not results[-1][1]:
            return _summary(results)

    results.append(("对齐坐标系", run_step(
        "第 1 步：原始数据 -> L21 固定坐标系",
        [python, ROOT / "scripts" / "align_h5_coordinates.py",
         "--input", raw_path, "--output", aligned_path],
    )))
    if not results[-1][1]:
        return _summary(results)

    results.append(("训练模型", run_step(
        "第 2 步：自监督训练（预测角度 -> FK -> 对比输入关键点）",
        [python, "-m", "retargeting", "train",
         "--input", aligned_path, "--run-name", RUN_NAME,
         "--epochs", args.epochs, "--batch-size", args.batch_size,
         "--device", args.device, "--checkpoint-root", checkpoint_root],
    )))
    if not results[-1][1]:
        return _summary(results)

    results.append(("导出角度", run_step(
        "第 3 步：用 checkpoint 导出 18 维 left/right_angles",
        [python, "-m", "retargeting", "export",
         "--input", aligned_path, "--checkpoint", checkpoint,
         "--output", angles_path, "--device", args.device],
    )))
    if not results[-1][1]:
        return _summary(results)

    banner("第 4 步：校验角度文件（结构 + 有效帧）")
    verified = angles_path.is_file() and verify_angles(angles_path)
    results.append(("校验角度文件", verified))
    if not verified:
        return _summary(results)

    results.append(("官方 inspect 校验", run_step(
        "第 4b 步：python -m retargeting inspect",
        [python, "-m", "retargeting", "inspect", "--angle-h5", angles_path],
    )))

    if args.no_replay:
        print("\n已按 --no-replay 跳过仿真回放。")
    else:
        results.append(("三机器人仿真回放", run_step(
            "第 5 步：三台机器人（H1-2 / GR1-T2 / G1）仿真回放",
            [python, ROOT / "scripts" / "show_hands_all.py", "--file", angles_path],
        )))

    return _summary(results, angles_path)


def _summary(results, angles_path=None):
    banner("流水线汇总")
    for title, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {title}")
    ok_all = all(item[1] for item in results) if results else False
    if angles_path is not None:
        print(f"\n  角度文件：{angles_path}")
    print(f"  结论：{'全链路跑通' if ok_all else '有步骤失败'}")
    print("=" * 88)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
