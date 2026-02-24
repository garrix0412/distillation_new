# 配置参考文档

本文档说明蒸馏管线中所有 YAML 配置项的含义、类型、默认值和调参建议。

---

## 1. 概览

所有配置文件位于 `configs/` 目录，使用 YAML 格式。命名规则：`<阶段>_d<码距>.yaml`。

每个配置文件包含以下部分：

| 部分 | 说明 | 必需 |
|------|------|------|
| `data` | 数据生成参数 | 是 |
| `model` | 学生模型架构 | 是 |
| `training` | 训练超参数 | 是 |
| `logging` | 日志和保存设置 | 是 |
| `teacher` | Teacher 加载配置 | 仅蒸馏 |
| `distillation` | 蒸馏损失权重 | 仅蒸馏 |

### 配置文件对照表

| 配置文件 | 管线阶段 | 训练脚本 |
|----------|---------|---------|
| `scratch_d3.yaml` | 从零训练（无蒸馏） | `train.py` |
| `mock_teacher_d3.yaml` | Mock Teacher 训练 | `train.py` |
| `baseline_kd_d3.yaml` | Baseline KD（仅响应） | `train_distill.py` |
| `stage1_kd_d3.yaml` | Stage 1（CNN + RNN） | `train_distill.py` |
| `stage1_v2_kd_d3.yaml` | Stage 1 v2（三路信号） | `train_distill.py` |
| `stage2_kd_d3.yaml` | Stage 2（fused logits） | `train_distill.py` |

---

## 2. `data` 部分

数据生成和加载参数。

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `distance` | int | 3 | 表面码码距 d，决定 stabilizer 数量 = d^2 - 1 |
| `rounds` | int | 5 | 纠错轮数（时序深度） |
| `noise_strength` | float | 0.005 | 物理错误率 p |
| `snr` | float | 10.0 | 软测量信噪比 |
| `use_soft` | bool | true | 是否使用软测量（模拟 I/Q 后验） |
| `num_train` | int | 200000 | 训练集样本数 |
| `num_val` | int | 20000 | 验证集样本数 |
| `batch_size` | int | 512 | 批大小 |
| `seed` | int | 42 | 随机种子 |
| `online` | bool | false | 是否每 epoch 重新采样训练数据 |

### 调参建议

- **distance**: d=3 用于快速验证，d=5/7 用于正式实验。增大 d 会显著增加 stabilizer 数量（d=3→8, d=5→24, d=7→48）。
- **noise_strength**: 典型范围 0.001-0.01。值越大，逻辑错误率越高，任务越难。
- **num_train**: 快速测试用 10000-50000，正式实验用 200000+。
- **seed**: Teacher 用 `seed=0`，Student 用 `seed=42`，确保数据不重叠。
- **online**: 推荐正式实验开启。离线模式适合快速迭代。

---

## 3. `model` 部分

学生模型架构参数。

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `size` | str | "small" | 模型尺寸预设 |
| `rnn_type` | str | "gru" | RNN 类型：`"gru"` 或 `"lstm"` |
| `init_checkpoint` | str | null | 初始化 checkpoint 路径（用于 Stage 2 从 Stage 1 初始化） |

### 模型尺寸对照表

| size | hidden_dim | conv_dim | readout_dim | CNN blocks | 约参数量 |
|------|-----------|----------|-------------|------------|---------|
| tiny | 16 | 8 | 8 | 1 | ~4K |
| small | 32 | 16 | 16 | 1 | ~16K |
| medium | 64 | 32 | 32 | 2 | ~80K |
| large | 128 | 64 | 64 | 2 | ~432K |

### 调参建议

- **size**: Student 推荐 `"small"` 或 `"medium"`。Teacher 使用 `"large"`。
- **rnn_type**: GRU 参数更少、训练更快；LSTM 对长序列可能更稳。d=3 推荐 GRU。
- **init_checkpoint**: Stage 2 必须设置为 Stage 1 的 best_model.pt 路径。

---

## 4. `teacher` 部分

Teacher 模型加载配置。仅 `train_distill.py` 使用。

### Mock Teacher（默认）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `type` | str | "mock" | Teacher 类型（可省略，默认 mock） |
| `checkpoint` | str | 必需 | Teacher checkpoint 路径 |
| `probe_heads` | str | null | Probe heads checkpoint 路径（Stage 2 需要） |

```yaml
# Mock teacher 示例（Baseline / Stage 1）
teacher:
  checkpoint: "checkpoints/mock_teacher_d3/best_model.pt"

# Mock teacher 示例（Stage 2 with probe heads）
teacher:
  checkpoint: "checkpoints/mock_teacher_d3/best_model.pt"
  probe_heads: "checkpoints/mock_teacher_d3/probe_heads.pt"
```

### 外部 Teacher（AlphaQubit）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `type` | str | 必需 | `"alphaqubit"` |
| `checkpoint` | str | 必需 | 外部模型 checkpoint 路径 |
| `hidden_dim` | int | 必需 | 外部模型隐藏维度 |
| `readout_dim` | int | 必需 | 外部模型 readout 维度 |

```yaml
# 外部 AlphaQubit teacher 示例
teacher:
  type: "alphaqubit"
  checkpoint: "path/to/alphaqubit_model.pt"
  hidden_dim: 256
  readout_dim: 128
```

### 向后兼容

无 `type` 字段时默认为 `"mock"`，所有现有配置无需修改。

详细适配器实现指南见 [docs/teacher_adapter_guide.md](teacher_adapter_guide.md)

---

## 5. `distillation` 部分

蒸馏损失权重和参数。总损失公式：

```
total = alpha * L_task + beta * L_response + gamma_cnn * L_cnn + gamma_rnn * L_rnn
        + gamma_readout * L_readout + gamma_fused * L_fused
```

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `alpha` | float | 0.5 | 任务损失权重（真实标签 BCE） |
| `beta` | float | 0.5 | 响应 KD 权重（soft logit 匹配） |
| `gamma_cnn` | float | 0.0 | CNN 特征 KD 权重（空间，Hook A） |
| `gamma_rnn` | float | 0.0 | RNN 特征 KD 权重（时序，Hook B） |
| `gamma_readout` | float | 0.0 | Readout 特征 KD 权重 |
| `gamma_fused` | float | 0.0 | Fused logit KD 权重（需 probe heads） |
| `temperature` | float | 1.0 | 蒸馏温度（>1 使分布更软） |
| `feature_loss_type` | str | "mse" | 特征损失类型：`"mse"` 或 `"cosine"` |

### 各阶段推荐权重配比

| 阶段 | alpha | beta | gamma_cnn | gamma_rnn | gamma_readout | gamma_fused |
|------|-------|------|-----------|-----------|---------------|-------------|
| Baseline | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Stage 1 | 0.3 | 0.3 | 0.2 | 0.2 | 0.0 | 0.0 |
| Stage 1 v2 | 0.25 | 0.25 | 0.15 | 0.15 | 0.2 | 0.0 |
| Stage 2 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 | 0.6 |

### 调参建议

- **temperature**: 推荐 2.0。过高（>5）可能使 teacher 信号过于均匀。
- **feature_loss_type**: MSE 更稳健，cosine 更关注方向。默认 MSE。
- 特征 KD（gamma_cnn/rnn）维度不匹配时自动添加线性投影层。

---

## 6. `training` 部分

训练超参数。

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `epochs` | int | 30 | 训练轮数 |
| `learning_rate` | float | 0.001 | 初始学习率 |
| `weight_decay` | float | 0.0001 | AdamW 权重衰减 |
| `scheduler` | str | "none" | 学习率调度器：`"none"` 或 `"cosine"` |
| `warmup_steps` | int | 0 | 线性 warmup 步数 |
| `grad_clip` | float | 0 | 梯度裁剪范数（0 为不裁剪） |
| `device` | str | "auto" | 设备：`"auto"`, `"cpu"`, `"cuda"`, `"mps"` |

### 调参建议

- **learning_rate**: Stage 2（微调）建议降低到 0.0005。
- **scheduler**: 推荐 cosine。
- **warmup_steps**: 100-200 步通常足够。
- **grad_clip**: 推荐 1.0，防止梯度爆炸。

---

## 7. `logging` 部分

日志和 checkpoint 保存设置。

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `log_interval` | int | 50 | 每 N 个 batch 打印训练日志 |
| `eval_interval` | int | 1 | 每 N 个 epoch 执行验证 |
| `save_dir` | str | 必需 | Checkpoint 保存目录 |

保存的文件：
- `best_model.pt` — 最佳验证 LER 的 checkpoint
- `final_model.pt` — 最终 epoch 的 checkpoint
- `config.yaml` — 训练使用的完整配置
- `history.json` — 训练历史记录

---

## 8. 常用 Recipes

### 快速测试（验证代码可运行）

```bash
# 使用 scratch_d3_quick.yaml 或手动缩小数据量
python train.py --config configs/scratch_d3_quick.yaml
```

### 完整 Mock Teacher 管线

```bash
# 一键执行全部 6 步
python run_pipeline.py

# 仅查看步骤
python run_pipeline.py --dry-run

# 从第 3 步恢复
python run_pipeline.py --from 3

# 跳过消融
python run_pipeline.py --skip-ablations
```

### 外部 Teacher 管线

```bash
# 使用外部 AlphaQubit teacher
python run_pipeline.py --teacher-mode external \
    --teacher-checkpoint path/to/alphaqubit.pt \
    --teacher-hidden-dim 256 \
    --teacher-readout-dim 128

# 预览步骤
python run_pipeline.py --teacher-mode external \
    --teacher-checkpoint path/to/alphaqubit.pt \
    --dry-run
```

### 单步手动执行

```bash
# 1. 训练 Mock Teacher
python train.py --config configs/mock_teacher_d3.yaml

# 2. 训练 Probe Heads
python train_probes.py --teacher_checkpoint checkpoints/mock_teacher_d3/best_model.pt

# 3. Baseline KD
python train_distill.py --config configs/baseline_kd_d3.yaml

# 4. Stage 1 v2 KD
python train_distill.py --config configs/stage1_v2_kd_d3.yaml

# 5. Stage 2 KD
python train_distill.py --config configs/stage2_kd_d3.yaml

# 6. 消融实验
python run_ablations.py
```
