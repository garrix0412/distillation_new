"""
端到端蒸馏管线运行器。

支持两种 Teacher 模式：
  - mock（默认）：训练 Mock Teacher → Probes → 多阶段 KD → 消融
  - external：使用外部 Teacher（如 AlphaQubit），跳过 teacher/probe 训练

Mock 模式完整管线步骤：
  1. 训练 mock teacher
  2. 训练 probe heads
  3. Stage 1 KD（CNN + RNN 特征 KD — Stage 2 和消融实验需要）
  4. Stage 1 v2 KD（三路信号：CNN + RNN + readout 特征 KD）
  5. Stage 2 KD（fused logits，从 Stage 1 v2 初始化）
  6. 消融实验

External 模式管线步骤：
  1. 训练 probe heads（在外部 teacher 中间特征上）
  2. Stage 1 v2 KD（三路信号，使用外部 teacher）
  3. Stage 2 KD（fused logits，从 Stage 1 v2 初始化）

用法：
    # Mock teacher 完整管线
    python run_pipeline.py
    python run_pipeline.py --from 3              # 从第 3 步恢复

    # 外部 teacher 管线
    python run_pipeline.py --teacher-mode external \\
        --teacher-checkpoint path/to/model.pt \\
        --teacher-hidden-dim 256 --teacher-readout-dim 128

    # 仅打印步骤
    python run_pipeline.py --dry-run
    python run_pipeline.py --teacher-mode external --dry-run

    # 指定码距
    python run_pipeline.py --distance 5

    # 跳过消融实验
    python run_pipeline.py --skip-ablations
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def get_mock_steps(distance, skip_ablations=False):
    """生成 mock teacher 模式的管线步骤。"""
    d = distance
    steps = [
        {
            "name": "Train mock teacher",
            "description": f"训练大号 StudentDecoder 作为 Mock Teacher（码距 d={d}）",
            "cmd": [sys.executable, "train.py",
                    "--config", f"configs/mock_teacher_d{d}.yaml"],
            "checkpoint": f"checkpoints/mock_teacher_d{d}/best_model.pt",
        },
        {
            "name": "Train probe heads",
            "description": "在冻结 Teacher 中间特征上训练辅助 probe heads，用于 Stage 2 fused logits",
            "cmd": [sys.executable, "train_probes.py",
                    "--config", f"configs/baseline_kd_d{d}.yaml",
                    "--save_dir", f"checkpoints/mock_teacher_d{d}"],
            "checkpoint": f"checkpoints/mock_teacher_d{d}/probe_heads.pt",
        },
        {
            "name": "Stage 1 KD (CNN + RNN feature KD)",
            "description": "CNN + RNN 双路特征蒸馏，为 Stage 2 和消融实验提供基础 checkpoint",
            "cmd": [sys.executable, "train_distill.py",
                    "--config", f"configs/stage1_kd_d{d}.yaml"],
            "checkpoint": f"checkpoints/stage1_kd_d{d}/best_model.pt",
        },
        {
            "name": "Stage 1 v2 KD (CNN + RNN + readout feature KD)",
            "description": "完整三路特征蒸馏：CNN + RNN + readout",
            "cmd": [sys.executable, "train_distill.py",
                    "--config", f"configs/stage1_v2_kd_d{d}.yaml"],
            "checkpoint": f"checkpoints/stage1_v2_kd_d{d}/best_model.pt",
        },
        {
            "name": "Stage 2 KD (fused logits, init from Stage 1 v2)",
            "description": "Fused logits 蒸馏，从 Stage 1 v2 checkpoint 初始化，使用 probe heads 提供融合信号",
            "cmd": [sys.executable, "train_distill.py",
                    "--config", f"configs/stage2_kd_d{d}.yaml"],
            "checkpoint": f"checkpoints/stage2_kd_d{d}/best_model.pt",
        },
    ]
    if not skip_ablations:
        steps.append({
            "name": "Ablation experiments",
            "description": "运行信号消融（Group A）和融合消融（Group B）实验矩阵",
            "cmd": [sys.executable, "run_ablations.py"],
            "checkpoint": None,
        })
    return steps


def apply_config_overrides(config_file, overrides):
    """
    加载原始 YAML 配置，深度合并 overrides，写入临时文件。

    Args:
        config_file: 原始 YAML 配置路径
        overrides: 要覆盖的字典（深度合并）

    Returns:
        临时配置文件路径
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    _deep_merge(config, overrides)

    tmp_dir = Path("checkpoints/_pipeline_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"tmp_{Path(config_file).name}"
    with open(tmp_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(tmp_path)


def _deep_merge(base, override):
    """将 override 字典深度合并入 base 字典（原地修改）。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_external_steps(distance, teacher_checkpoint, teacher_type="alphaqubit",
                       teacher_hidden_dim=256, teacher_readout_dim=128,
                       skip_ablations=False):
    """
    生成外部 teacher 模式的管线步骤。

    跳过 teacher 训练，但包含 probe 训练步骤（需要外部 teacher 返回 cnn_features 和 decoder_states）。
    自动将外部 teacher 配置注入到现有 YAML 中。
    """
    d = distance

    teacher_overrides = {
        "teacher": {
            "type": teacher_type,
            "checkpoint": teacher_checkpoint,
            "hidden_dim": teacher_hidden_dim,
            "readout_dim": teacher_readout_dim,
        }
    }

    probe_save_dir = f"checkpoints/external_teacher_d{d}"

    # Probe 训练：复用 baseline KD 配置，覆盖 teacher 部分
    probe_config = f"configs/baseline_kd_d{d}.yaml"
    if Path(probe_config).exists():
        probe_tmp = apply_config_overrides(probe_config, teacher_overrides)
    else:
        probe_tmp = probe_config

    # Stage 1 v2: 三路特征 KD
    stage1v2_config = f"configs/stage1_v2_kd_d{d}.yaml"
    if Path(stage1v2_config).exists():
        stage1v2_tmp = apply_config_overrides(stage1v2_config, teacher_overrides)
    else:
        stage1v2_tmp = stage1v2_config

    # Stage 2: fused logits KD（使用 probe heads）
    stage2_config = f"configs/stage2_kd_d{d}.yaml"
    stage2_overrides = {
        **teacher_overrides,
        "teacher": {
            **teacher_overrides["teacher"],
            "probe_heads": f"{probe_save_dir}/probe_heads.pt",
        },
    }
    if Path(stage2_config).exists():
        stage2_tmp = apply_config_overrides(stage2_config, stage2_overrides)
    else:
        stage2_tmp = stage2_config

    steps = [
        {
            "name": "Train probe heads (external teacher)",
            "description": f"在外部 {teacher_type} teacher 中间特征上训练 probe heads",
            "cmd": [sys.executable, "train_probes.py",
                    "--config", probe_tmp,
                    "--save_dir", probe_save_dir],
            "checkpoint": f"{probe_save_dir}/probe_heads.pt",
        },
        {
            "name": "Stage 1 v2 KD (external teacher)",
            "description": f"使用外部 {teacher_type} teacher 进行三路特征蒸馏",
            "cmd": [sys.executable, "train_distill.py",
                    "--config", stage1v2_tmp],
            "checkpoint": f"checkpoints/stage1_v2_kd_d{d}/best_model.pt",
        },
        {
            "name": "Stage 2 KD (external teacher)",
            "description": f"使用外部 {teacher_type} teacher 进行 fused logits 蒸馏（从 Stage 1 v2 初始化）",
            "cmd": [sys.executable, "train_distill.py",
                    "--config", stage2_tmp],
            "checkpoint": f"checkpoints/stage2_kd_d{d}/best_model.pt",
        },
    ]

    if not skip_ablations:
        steps.append({
            "name": "Ablation experiments",
            "description": "运行信号消融实验矩阵",
            "cmd": [sys.executable, "run_ablations.py"],
            "checkpoint": None,
        })

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="端到端蒸馏管线运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用示例：
  # Mock teacher 完整管线
  python run_pipeline.py

  # 从第 3 步恢复
  python run_pipeline.py --from 3

  # 外部 teacher 管线
  python run_pipeline.py --teacher-mode external \\
      --teacher-checkpoint path/to/model.pt \\
      --teacher-hidden-dim 256 --teacher-readout-dim 128

  # 仅打印步骤（不执行）
  python run_pipeline.py --dry-run

  # 指定码距 d=5（需要对应 configs/*_d5.yaml）
  python run_pipeline.py --distance 5

  # 跳过消融实验
  python run_pipeline.py --skip-ablations

配置文件说明：
  configs/mock_teacher_d3.yaml    Mock Teacher 训练配置
  configs/stage1_kd_d3.yaml       Stage 1 KD（CNN + RNN 特征）
  configs/stage1_v2_kd_d3.yaml    Stage 1 v2 KD（三路信号）
  configs/stage2_kd_d3.yaml       Stage 2 KD（fused logits）
  docs/config_guide.md            完整配置参考文档
""",
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="从第 N 步开始（1-indexed，默认: 1）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印管线步骤，不实际执行",
    )
    parser.add_argument(
        "--teacher-mode", choices=["mock", "external"], default="mock",
        help="Teacher 模式: mock=训练 Mock Teacher（默认）, external=使用外部模型",
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default=None,
        help="外部 teacher checkpoint 路径（仅 --teacher-mode external）",
    )
    parser.add_argument(
        "--teacher-type", type=str, default="alphaqubit",
        help="外部 teacher 类型（默认: alphaqubit）",
    )
    parser.add_argument(
        "--teacher-hidden-dim", type=int, default=256,
        help="外部 teacher 隐藏维度（默认: 256）",
    )
    parser.add_argument(
        "--teacher-readout-dim", type=int, default=128,
        help="外部 teacher readout 维度（默认: 128）",
    )
    parser.add_argument(
        "--distance", type=int, default=3,
        help="表面码码距（默认: 3），影响配置文件选择",
    )
    parser.add_argument(
        "--skip-ablations", action="store_true",
        help="跳过消融实验步骤",
    )
    args = parser.parse_args()

    # 校验 external 模式参数
    if args.teacher_mode == "external" and args.teacher_checkpoint is None:
        parser.error("--teacher-mode external 需要指定 --teacher-checkpoint")

    # 生成管线步骤
    if args.teacher_mode == "mock":
        steps = get_mock_steps(args.distance, skip_ablations=args.skip_ablations)
    else:
        steps = get_external_steps(
            distance=args.distance,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_type=args.teacher_type,
            teacher_hidden_dim=args.teacher_hidden_dim,
            teacher_readout_dim=args.teacher_readout_dim,
            skip_ablations=args.skip_ablations,
        )

    # 显示管线步骤
    mode_label = f"mock (d={args.distance})" if args.teacher_mode == "mock" \
        else f"external/{args.teacher_type} (d={args.distance})"
    print(f"{'='*60}")
    print(f"Distillation Pipeline — {mode_label}")
    print(f"{'='*60}")

    for i, step in enumerate(steps, 1):
        status = "SKIP" if i < args.from_step else "RUN"
        print(f"  Step {i}: {step['name']} [{status}]")
        if args.dry_run:
            print(f"         {step['description']}")
            print(f"         cmd: {' '.join(step['cmd'])}")
    print()

    if args.dry_run:
        print("Dry run — no commands executed.")
        return

    # 执行管线
    for i, step in enumerate(steps, 1):
        if i < args.from_step:
            continue

        # 如果 checkpoint 已存在则跳过
        if step["checkpoint"] and Path(step["checkpoint"]).exists():
            print(f"[Step {i}/{len(steps)}] {step['name']} — checkpoint exists, skipping")
            continue

        print(f"[Step {i}/{len(steps)}] {step['name']}")
        print(f"  {step['description']}")
        print(f"  Command: {' '.join(step['cmd'])}")

        result = subprocess.run(step["cmd"])
        if result.returncode != 0:
            print(f"\nERROR: Step {i} failed (exit code {result.returncode})")
            print(f"Fix the issue and resume with: python run_pipeline.py --from {i}")
            sys.exit(1)

        print(f"  Done.\n")

    # 清理临时配置
    tmp_dir = Path("checkpoints/_pipeline_tmp")
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
