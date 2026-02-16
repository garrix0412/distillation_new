# AlphaQubit Teacher 蒸馏计划
## 1) 目标与范围
- 以 AlphaQubit 复现模型作为 **Teacher**，通过离线蒸馏训练更小的 **Student neural decoder**。  


---

## 2) 总体路线概览
路线：**Baseline（只蒸最终 logits）→ 分步蒸馏 Stage 1（三路监督联合）→ 构造 fused logits（logit 空间融合目标）→ Stage 2（同一 Student 的二阶段蒸馏）→ 系统性消融实验验证贡献与必要性**。

约束：不引入额外推理模型；融合/多出口/投影头等仅在训练期用于构造软监督，最终推理仅保留单一 Student。

---

## 3) 任务 1：Baseline KD（Offline Response-Based KD）
**任务说明**  
- 采用标准离线输出蒸馏：使用 Teacher 最终输出的 **soft logits / posterior** 监督 Student 的最终输出（I/X/Z/Y 或 X/Z 两通道）。

**任务目的**  
- 建立统一对照基线，量化“只蒸最终输出”能够达到的压缩效果与可达到水平，为后续分步蒸馏与融合目标提供参照。

**任务清单**
- [ ] 固定 Teacher（参数冻结），明确 Teacher/Student 输出形式（I/X/Z/Y 或 X/Z）。  
- [ ] 实现 baseline 蒸馏训练：Teacher(final logits) → Student(final logits)（KL/CE 型蒸馏损失），同时保留任务监督（GT）。  
- [ ] 跑通训练流程并保存 baseline student checkpoint；保存 baseline 训练配置（保证可复现）。

**阶段结果**  
- 得到“仅使用 final-logit 蒸馏”的 baseline Student，作为后续所有方案的主要对照组。

---

## 4) 任务 2：分步蒸馏 Stage 1（多信号联合监督）
**任务说明**  
在同一个 Student 上引入三路监督信号，使 Student 不仅拟合最终输出，还学习 Teacher 的内部建模信息：

1) **Transformer feature-KD（空间相关）**  
- 对齐 Teacher 在 Transformer（空间混合）相关模块产生的中间表征（feature/hidden states）。

2) **RNN feature-KD（时间记忆）**  
- 对齐 Teacher 在 recurrent/RNN（时间混合）相关模块产生的中间表征。

3) **Readout response-KD（决策映射）**  
- 对齐 Teacher readout 输出 logits，蒸馏“内部表示 → 最终决策”的映射规律。

**任务目的**  
- 为小模型提供更密集的训练信号，使空间、时间与读出决策的信息能够被显式学习，从而提高可压缩性与稳定性。

**任务清单**
- [ ] 定义 Student 与 Teacher 的“中间表征接口”（用于 feature-KD 对齐）。  
- [ ] 需要时加入轻量投影/adapter（仅用于维度对齐，不改变推理结构）。  
- [ ] 将三路 loss 与 baseline final-logit KD / task loss 组合为联合训练目标，完成 Stage 1 训练。  
- [ ] 保存 Stage 1 student checkpoint；保存 Stage 1 配置，支持三路信号分别开关。

**阶段结果**  
- 得到完成“多信号联合学习”的 Stage 1 Student，作为 Stage 2 的初始化起点，同时可直接用于信号开关消融。

---

## 5) 任务 3：构造 fused logits（logit 空间融合目标）
**任务说明**  
将 Teacher 不同layer的信息统一映射到 **logit 空间**，形成多个 logits 输出并进行融合，得到 **fused soft target**：\(z_{\text{fuse}}\)。

典型三路 logits 来源：
- 来自 Transformer 路的 logits（将 Transformer 中间表征映射到输出空间）  
- 来自 RNN 路的 logits（将 RNN 中间表征映射到输出空间）  
- 来自最终 readout 的 logits（Teacher 原生最终输出）

融合规则在第一版采用简单形式（如均值/固定权重），复杂融合策略作为后续扩展任务。

**任务目的**  
- 将“空间/时间/读出”三类信息在输出层面合成为一个统一软目标，为二阶段蒸馏提供更综合、更易对齐的监督信号。

**任务清单**
- [ ] 在 Teacher 侧构造三个位置的 logits 输出（训练期使用的轻量映射/探针头）。  
- [ ] 定义融合规则生成 \(z_{\text{fuse}}\)（第一版使用均值或固定权重）。  
- [ ] 将 fused logits 生成过程封装为可复用模块（训练时可直接调用）。

**阶段结果**  
- 得到稳定可复用的 \(z_{\text{fuse}}\) 软目标生成流程，用于后续 Stage 2 训练以及“融合相关消融”。

---

## 6) 任务 4：分步蒸馏 Stage 2（同一 Student 的二阶段训练）
**任务说明**  
在同一 Student 上进行第二阶段训练，以 fused logits \(z_{\text{fuse}}\) 作为主要蒸馏目标，使 Student 输出收敛到“融合后的 teacher 目标”。

Stage 2 的典型做法：
- 以 Stage 1 student 作为初始化  
- 以 fused-logit KD 为主  
- 视训练稳定性，弱化或关闭 feature-KD，仅保留必要的 task loss / 少量输出 KD

**任务目的**  
- 将 Stage 1 学到的内部建模信息进一步压缩到输出层面，提升更小模型的可蒸馏性与最终收敛质量。

**任务清单**
- [ ] 以 Stage 1 checkpoint 初始化 Student。  
- [ ] 进行 Stage 2 训练：主损失为 fused-logit KD，调整/关闭 feature-KD（保留必要的稳定项）。  
- [ ] 保存最终 student checkpoint；保存 Stage 2 训练配置。

**阶段结果**  
- 得到两阶段训练后的最终 Student（完整方案版本），同时获得可复现的 Stage 2 训练流程。

---

## 7) 任务 5：消融实验设计
分步蒸馏与融合蒸馏天然包含四类可开关的蒸馏信号：  
- Transformer feature 蒸馏（空间相关表征）  
- RNN feature 蒸馏（时间记忆表征）  
- Readout response 蒸馏（读出映射/最终决策映射）  
- fused-logit 蒸馏（logit 空间融合目标，用于 Stage 2）

因此消融实验围绕“**信号开关**”与“**训练阶段（单阶段/两阶段）**”两条主轴组织。这样组织的优点是：每条改动都有明确含义，结果可解释性强，同时避免全组合带来的实验数量爆炸。

### 5.1 分步蒸馏信号消融（回答：哪一路有效、是否互补）
**可设计的对照类型**
- 基础对照：无蒸馏 vs 仅最终 logits 的离线输出蒸馏（baseline）。  
- 单信号增量：在 baseline 上分别加入  
  - Transformer feature 蒸馏；  
  - RNN feature 蒸馏；  
  - Readout response 蒸馏。  
- 组合增量：在 baseline 上同时加入（至少覆盖）  
  - Transformer + RNN（空间+时间）；  
  - Transformer + RNN + Readout（三路分步信号全开）。

**为什么这样设计**
- “baseline → 单信号 → 组合信号”的结构能把增益分解成**主效应**（每一路单独贡献）与**互补效应**（空间与时间是否叠加、读出蒸馏是否提供额外约束）。  
- 保持其他训练条件一致时，差异更容易归因到“新增的蒸馏信号”，便于解释“哪些监督真正有用”。

**好处**
- 快速定位：空间蒸馏、时间蒸馏、读出蒸馏中，哪一项对小模型最关键。  
- 判断冗余与互补：若组合明显优于单项，说明信号互补；若组合提升很小，说明信息可能重叠或对齐方式需要调整。  
- 为后续资源投入提供方向：把训练预算集中在贡献最大的信号与组合上。

### 5.2 融合目标与两阶段训练消融
**可设计的对照类型**
- 不融合 vs 融合：  
  - “只用最终 readout logits 做蒸馏”对比“使用 fused logits 做蒸馏”。  
- 单阶段 vs 两阶段：  
  - 仅进行 Stage 1（分步信号联合学习）对比 Stage 1 → Stage 2（加入 fused-logit 收敛）。  
- 融合蒸馏的独立性与叠加性：  
  - 仅使用 fused-logit 进行蒸馏（验证融合目标本身是否有效）对比  
  - 先分步学习再 fused（验证两阶段是否带来更强结果/更稳定收敛）。

**为什么这样设计**
- fused-logit 蒸馏与分步 feature/读出蒸馏的角色不同：  
  - 分步信号偏“让中间机制可学习”；  
  - fused 目标偏“把信息压到统一输出空间，利于更小模型收敛”。  
- 因此需要把“融合目标的独立贡献”和“融合目标是否依赖 Stage 1 的前置学习”拆开验证，否则难以判断 Stage 2 的价值来自哪里。

**好处**
- 明确核心路线的必要性：两阶段训练到底是“必须步骤”还是“可选优化”。  
- 若 fused 目标单独就有效，可直接简化主线；若必须依赖 Stage 1，说明“先学机制再收敛”的流程更合理。  
- 形成清晰结论链：分步信号负责学习什么、融合目标负责压缩什么，便于后续汇报与推进。


