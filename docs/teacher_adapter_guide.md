# Teacher 适配器集成指南


---

## 1. 概览

### 1.1 TeacherAdapter 是什么

`TeacherAdapter`（定义于 `models/teacher.py`）是所有 Teacher 模型的抽象基类。蒸馏管线通过此接口统一访问 Teacher 的预测结果和中间表征，无需关心底层模型的具体实现。

### 1.2 整体流程

```
YAML 配置
   │
   ▼
load_teacher(config, distance, device)    ← 统一入口，按 type 分派
   │
   ▼
TeacherAdapter 实例（已冻结参数）
   │
   ▼
forward_with_intermediates(inputs)        ← 蒸馏训练循环中调用
   │
   ├── logits:         [B, 1]             ← 直接用于 ResponseKDLoss
   └── intermediates:  dict               ← 各键分别用于 FeatureKDLoss
          │
          ▼
     DistillationLoss.forward()            ← 按 gamma 权重选择性消费
```

### 1.3 需实现的接口

| 成员 | 类型 | 说明 |
|------|------|------|
| `forward_with_intermediates(inputs)` | 方法 | 返回 `(logits, intermediates)` |
| `hidden_dim` | 属性 | CNN/RNN 特征最后一维的大小 |
| `readout_dim` | 属性 | readout 特征最后一维的大小 |

---

## 2. 输入规范（inputs dict）

`forward_with_intermediates` 接收的 `inputs` 是由 `SurfaceCodeDataset.__getitem__` 产生、经 DataLoader batching 后的字典。

### 2.1 键和张量格式

| 键 | Shape | Dtype | 值域 | 说明 |
|----|-------|-------|------|------|
| `detection_events` | `[B, rounds, n_stab]` | float32 | {0.0, 1.0} | 硬检测事件（始终存在） |
| `soft_events` | `[B, rounds, n_stab]` | float32 | (0, 1) | 软测量后验概率（仅 `use_soft=true` 时存在） |

### 2.2 码距与 stabilizer 数量

```
n_stab = d² - 1

d=3  → n_stab=8
d=5  → n_stab=24
d=7  → n_stab=48
```

### 2.3 外部模型的格式转换

如果外部模型的输入格式不同（如期望 `[B, n_stab, rounds]` 或 int8），应在 adapter 的 `forward_with_intermediates` 内部完成转换，不要修改管线代码。

---

## 3. 输出规范（intermediates dict 契约）

### 3.1 所有可用键

| 键 | Shape | 对应损失权重 | 说明 |
|----|-------|-------------|------|
| `cnn_features` | `[B, rounds, n_stab, hidden_dim]` | `gamma_cnn` | 空间特征（Hook A） |
| `decoder_states` | `[B, rounds, n_stab, hidden_dim]` | `gamma_rnn` | 时序特征（Hook B） |
| `readout_features` | `[B, readout_dim]` | `gamma_readout` | readout 内部特征 |
| `readout_logits` | `[B, 1]` | `beta`（间接） | 输出 logits（Hook C/D） |

### 3.2 哪些键是必需的

**只需返回你打算使用的信号对应的键。**

`DistillationLoss.forward()` 的访问逻辑（摘自 `distillation/losses.py`）：

```python
# gamma_cnn > 0 时才访问 cnn_features
if self.gamma_cnn > 0 and student_intermediates is not None and teacher_intermediates is not None:
    l_cnn = self.cnn_feature_kd(
        student_intermediates["cnn_features"],
        teacher_intermediates["cnn_features"],      # ← 这里访问
    )

# gamma_rnn > 0 时才访问 decoder_states
if self.gamma_rnn > 0 and ...:
    l_rnn = self.rnn_feature_kd(
        student_intermediates["decoder_states"],
        teacher_intermediates["decoder_states"],     # ← 这里访问
    )

# gamma_readout > 0 时才访问 readout_features
if self.gamma_readout > 0 and ...:
    l_readout = self.readout_feature_kd(
        student_intermediates["readout_features"],
        teacher_intermediates["readout_features"],   # ← 这里访问
    )
```

因此：
- 如果你的外部模型**没有 CNN 层**，将配置中 `gamma_cnn: 0.0`，无需返回 `cnn_features`
- 如果你的外部模型**没有 RNN 层**，将配置中 `gamma_rnn: 0.0`，无需返回 `decoder_states`
- `readout_logits` 用于 `ResponseKDLoss`（`beta > 0` 时）和 probe heads，通常都需要返回

### 3.3 Probe Heads 相关的可选键

以下键由 `ProbeHeadSet` 自动产生，adapter 无需关心：

| 键 | 说明 |
|----|------|
| `fused_logits` | 三路融合 logit（`gamma_fused` 使用） |
| `fused_cnn_logits` | CNN probe logit |
| `fused_rnn_logits` | RNN probe logit |

### 3.4 通用约束

- 所有返回张量必须在**同一 device** 上
- dtype 为 **float32**
- adapter 的 `forward_with_intermediates` 应使用 `@torch.no_grad()` 装饰
- 所有输出张量应 `.detach()`（防止梯度流向 teacher）
- `rounds` 和 `n_stab` 维度 **必须** 与 student 一致（损失函数做逐元素对比）

---

## 4. 维度匹配与自动投影

### 4.1 hidden_dim 属性

`hidden_dim` 对应 `cnn_features` 和 `decoder_states` 张量的最后一维。当 student 的 `hidden_dim` 与 teacher 的 `hidden_dim` 不同时，`FeatureKDLoss` 会自动插入 `nn.Linear(student_dim, teacher_dim)` 投影层。

### 4.2 readout_dim 属性

`readout_dim` 对应 `readout_features` 张量的最后一维。同样支持自动投影。

### 4.3 不支持自动对齐的维度

`rounds` 和 `n_stab` 维度**不会**被自动对齐。如果外部模型的这两个维度与 student 不同，你需要在 adapter 内部手动对齐（如截断、填充或插值）。

### 4.4 投影层参数

投影层的参数会自动加入优化器参与训练，无需手动处理。

---

## 5. Probe Heads 兼容性

Probe heads 是在冻结的 Teacher 特征上预训练的辅助 readout 头（`ProbeHeadSet`），用于 Stage 2 的 fused logits 蒸馏。

**前提条件：**

- 外部 Teacher 若要使用 probe heads，其 adapter 必须在 `intermediates` 中返回 `cnn_features` 和 `decoder_states`
- `ProbeHeadSet.forward()` 会访问 `cnn_features[:, -2, :, :]` 和 `decoder_states[:, -2, :, :]`（倒数第二轮）
- `train_probes.py` 通过 `--config` YAML 配置 + `load_teacher()` 统一入口加载任意类型 Teacher
- `run_pipeline.py --teacher-mode external` 已自动包含 probe head 训练步骤并在 Stage 2 启用 fused logits
- 如果外部模型不返回 `cnn_features` / `decoder_states`，probe 训练不可用，应将 `gamma_fused: 0.0`

---

## 6. 分步实现指南

### Step 1: 创建 adapter 文件

复制 `AlphaQubitAdapter` 模板作为起点：

```python
# models/my_teacher_adapter.py
import torch
import torch.nn as nn
from models.teacher import TeacherAdapter


class MyTeacherAdapter(nn.Module, TeacherAdapter):
    """自定义 Teacher 适配器。"""

    def __init__(self, checkpoint_path, hidden_dim, readout_dim, distance, device="cpu"):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._readout_dim = readout_dim
        # Step 2 在这里实现
```

### Step 2: 实现 `__init__`

加载模型、冻结参数、移到目标设备：

```python
    def __init__(self, checkpoint_path, hidden_dim, readout_dim, distance, device="cpu"):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._readout_dim = readout_dim

        # 加载你的模型
        self.model = YourModel.load(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        # 冻结所有参数（Teacher 不参与梯度更新）
        for param in self.model.parameters():
            param.requires_grad = False
```

### Step 3: 实现 `forward_with_intermediates`

将模型输出映射到标准 intermediates 格式：

```python
    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        self.model.eval()

        # 获取输入（如需格式转换，在此处理）
        x = inputs["detection_events"]  # [B, rounds, n_stab]

        # 调用你的模型（根据你的模型 API 调整）
        output = self.model(x)

        # 构建 intermediates dict
        logits = output["logits"]               # [B, 1]
        intermediates = {
            "readout_logits": logits.detach(),   # [B, 1]（通常必需）
        }

        # 按需添加中间表征（只添加你打算在 YAML 中启用的信号）
        if "spatial_features" in output:
            intermediates["cnn_features"] = output["spatial_features"].detach()
            # 确保 shape 为 [B, rounds, n_stab, hidden_dim]

        if "temporal_states" in output:
            intermediates["decoder_states"] = output["temporal_states"].detach()
            # 确保 shape 为 [B, rounds, n_stab, hidden_dim]

        if "readout_features" in output:
            intermediates["readout_features"] = output["readout_features"].detach()
            # 确保 shape 为 [B, readout_dim]

        return logits.detach(), intermediates
```

### Step 4: 设置 `hidden_dim` / `readout_dim`

```python
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def readout_dim(self) -> int:
        return self._readout_dim
```

### Step 5: 注册到 `load_teacher`

在 `models/teacher.py` 的 `load_teacher()` 函数中添加新分支：

```python
elif teacher_type == "my_teacher":
    from .my_teacher_adapter import MyTeacherAdapter
    return MyTeacherAdapter(
        checkpoint_path=teacher_config["checkpoint"],
        hidden_dim=teacher_config["hidden_dim"],
        readout_dim=teacher_config["readout_dim"],
        distance=distance,
        device=device,
    )
```

### Step 6: 编写 YAML 配置

```yaml
teacher:
  type: "my_teacher"
  checkpoint: "path/to/your_model.pt"
  hidden_dim: 256    # 你的模型的隐藏维度
  readout_dim: 128   # 你的模型的 readout 维度

distillation:
  alpha: 0.3          # 任务损失
  beta: 0.3           # 响应 KD
  gamma_cnn: 0.2      # CNN 特征 KD（没有 CNN 则设 0）
  gamma_rnn: 0.2      # RNN 特征 KD（没有 RNN 则设 0）
  gamma_readout: 0.0  # readout 特征 KD
  gamma_fused: 0.0    # 需先训练 probe heads 才能启用
  temperature: 2.0
```

### Step 7: 单独测试 adapter

在接入管线前，先独立验证 adapter 的输入输出格式：

```python
import torch
from models.teacher import load_teacher

# 模拟加载
teacher_config = {
    "type": "my_teacher",
    "checkpoint": "path/to/your_model.pt",
    "hidden_dim": 256,
    "readout_dim": 128,
}
teacher = load_teacher(teacher_config, distance=3, device="cpu")

# 构造模拟输入
B, R, N = 4, 5, 8  # batch=4, rounds=5, n_stab=8 (d=3)
inputs = {
    "detection_events": torch.randint(0, 2, (B, R, N)).float(),
    "soft_events": torch.rand(B, R, N),
}

# 前向传播
logits, intermediates = teacher.forward_with_intermediates(inputs)

# 验证输出格式
print(f"logits shape: {logits.shape}")               # 期望: [4, 1]
print(f"logits requires_grad: {logits.requires_grad}")  # 期望: False

for key, val in intermediates.items():
    print(f"{key}: shape={val.shape}, requires_grad={val.requires_grad}")
    # cnn_features:      期望 [4, 5, 8, 256], requires_grad=False
    # decoder_states:    期望 [4, 5, 8, 256], requires_grad=False
    # readout_features:  期望 [4, 128],       requires_grad=False
    # readout_logits:    期望 [4, 1],          requires_grad=False

# 验证维度属性
print(f"hidden_dim: {teacher.hidden_dim}")   # 期望: 256
print(f"readout_dim: {teacher.readout_dim}")  # 期望: 128
```

### Step 8: 接入管线运行

```bash
# 直接使用 train_distill.py
python train_distill.py --config configs/your_kd_config.yaml

# 或使用 run_pipeline.py（外部 teacher 模式）
python run_pipeline.py --teacher-mode external \
    --teacher-checkpoint path/to/your_model.pt \
    --teacher-hidden-dim 256 \
    --teacher-readout-dim 128
```

---

## 7. 调试清单

### Shape mismatch

**现象**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`

**常见原因与解法**:
- `rounds` 不一致：确认 adapter 输出的 `cnn_features` / `decoder_states` 的第二维等于配置中的 `data.rounds`
- `n_stab` 不一致：确认第三维等于 `d² - 1`（由 `data.distance` 决定）
- `hidden_dim` 不一致：检查 `teacher.hidden_dim` 属性是否与张量最后一维匹配（此处 `FeatureKDLoss` 会自动投影 student 侧，但 teacher 侧必须自洽）

### Device mismatch

**现象**: `RuntimeError: Expected all tensors to be on the same device`

**解法**: 确保 adapter 返回的所有张量都在传入的 `device` 上。常见遗漏：模型加载时未 `.to(device)`，或中间计算在 CPU 上。

### 梯度泄漏

**现象**: Teacher 参数意外出现梯度，训练变慢或结果异常。

**解法**:
1. `forward_with_intermediates` 加 `@torch.no_grad()` 装饰器
2. 所有输出张量调用 `.detach()`
3. `__init__` 中对所有参数设 `param.requires_grad = False`

### KeyError

**现象**: `KeyError: 'cnn_features'`（或其他键）

**原因**: 配置中 `gamma_cnn > 0` 但 adapter 未返回 `cnn_features`。

**解法**: 要么在 adapter 中返回该键，要么将对应的 gamma 设为 `0.0`。

### 维度属性不匹配

**现象**: `FeatureKDLoss` 的投影层输出维度与 teacher 张量不匹配。

**原因**: `teacher.hidden_dim` 属性值与 `cnn_features` / `decoder_states` 的实际最后一维大小不一致。

**解法**: 确保属性返回值与实际张量维度严格一致。

---

## 8. 完整代码示例

以下是一个可运行的 adapter 骨架，用随机张量模拟外部模型输出。你可以用它直接配合 `train_distill.py` 测试管线是否正常工作。

```python
"""
示例 adapter：用随机张量模拟外部 Teacher。
仅供测试管线流程，不产生有意义的蒸馏信号。

使用方法：
  1. 将此文件保存为 models/dummy_adapter.py
  2. 在 models/teacher.py 的 load_teacher() 中添加分支（见 Step 5）
  3. 配置 YAML 中设 type: "dummy"
"""

import torch
import torch.nn as nn
from models.teacher import TeacherAdapter


class DummyTeacherAdapter(nn.Module, TeacherAdapter):
    """用随机输出模拟外部 Teacher，用于测试管线流程。"""

    def __init__(self, hidden_dim, readout_dim, distance, device="cpu"):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._readout_dim = readout_dim
        self._distance = distance
        self._n_stab = distance ** 2 - 1
        self._device = device

    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        det = inputs["detection_events"]  # [B, rounds, n_stab]
        B, R, N = det.shape
        device = det.device

        # 模拟输出（实际 adapter 中替换为真实模型调用）
        logits = torch.randn(B, 1, device=device)

        intermediates = {
            "cnn_features": torch.randn(B, R, N, self._hidden_dim, device=device),
            "decoder_states": torch.randn(B, R, N, self._hidden_dim, device=device),
            "readout_features": torch.randn(B, self._readout_dim, device=device),
            "readout_logits": logits,
        }

        return logits, intermediates

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def readout_dim(self) -> int:
        return self._readout_dim
```

在 `models/teacher.py` 的 `load_teacher()` 中注册：

```python
elif teacher_type == "dummy":
    from .dummy_adapter import DummyTeacherAdapter
    return DummyTeacherAdapter(
        hidden_dim=teacher_config.get("hidden_dim", 64),
        readout_dim=teacher_config.get("readout_dim", 32),
        distance=distance,
        device=device,
    )
```

对应 YAML 配置：

```yaml
teacher:
  type: "dummy"
  checkpoint: "unused"   # DummyAdapter 不需要 checkpoint
  hidden_dim: 64
  readout_dim: 32

distillation:
  alpha: 0.3
  beta: 0.3
  gamma_cnn: 0.2
  gamma_rnn: 0.2
  gamma_readout: 0.0
  gamma_fused: 0.0
  temperature: 2.0
```
