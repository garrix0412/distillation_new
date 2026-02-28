# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge distillation pipeline for quantum error correction (QEC) neural decoders on the rotated surface code. A large "Teacher" model (mock AlphaQubit) is distilled into a smaller "Student" model designed for FPGA deployment. The project follows a multi-stage distillation plan: Baseline KD → Stage 1 (multi-signal) → Stage 2 (fused logits).

## Commands

```bash
# Train student from scratch (no distillation)
python train.py --config configs/scratch_d3.yaml

# Train mock teacher (large model used as teacher)
python train.py --config configs/mock_teacher_d3.yaml

# Baseline KD (soft logit matching only)
python train_distill.py --config configs/baseline_kd_d3.yaml

# Stage 1 KD (multi-signal: response + CNN feature + RNN feature)
python train_distill.py --config configs/stage1_kd_d3.yaml

# Stage 1 v2 KD (three-signal: CNN + RNN + readout feature KD)
python train_distill.py --config configs/stage1_v2_kd_d3.yaml

# Train probe heads on frozen teacher features
python train_probes.py --config configs/baseline_kd_d3.yaml
python train_probes.py --config configs/baseline_kd_d3.yaml --save_dir checkpoints/probes/
python train_probes.py --config configs/baseline_kd_d3.yaml --epochs 20 --lr 0.0005

# End-to-end pipeline (Teacher → Probes → Stage 1 v2 → Stage 2 → Ablations)
python run_pipeline.py              # Full pipeline
python run_pipeline.py --from 3     # Resume from step 3
python run_pipeline.py --dry-run    # Preview steps without executing

# External teacher pipeline (Probes → Stage 1 v2 → Stage 2)
python run_pipeline.py --teacher-mode external \
    --teacher-checkpoint path/to/model.pt \
    --teacher-hidden-dim 256 --teacher-readout-dim 128

# AlphaQubitV2 teacher KD (response + readout feature only)
python train_distill.py --config configs/alphaqubit_kd_d3.yaml
```

Dependencies: `torch`, `stim`, `numpy`, `pyyaml`. Install via `pip install -r requirements.txt`.

## Architecture

### Model Pipeline (per round)
`StabilizerEmbedder → DecoderRNN (GRU/LSTM) → CNNBlock (dilated 2D conv) → update state`
After all rounds: `ReadoutNetwork → logit`

The Student processes syndrome data **round-by-round** recurrently. Each round: embed stabilizer measurements, RNN update with previous state, CNN spatial mixing on a 2D grid. After the final round, a readout network mean-pools over stabilizers and predicts P(logical error) as a single logit (binary classification with BCEWithLogitsLoss).

### Key Design Decisions
- **CNN replaces Transformer**: Dilated 2D convolutions replace AlphaQubit's Syndrome Transformer for FPGA friendliness. Stabilizers are scattered to a `(d+1)x(d+1)` grid, convolved, then gathered back.
- **Per-stabilizer RNN**: GRUCell/LSTMCell operates on flattened `[batch * n_stab, hidden_dim]` — each stabilizer has independent temporal state.
- **Soft readout**: Simulated analog I/Q measurement posteriors are generated from hard detection events via Bayesian inference with a safety clamp (soft values never cross the 0.5 boundary).

### Teacher Loading Interface
All teachers implement `TeacherAdapter` ABC (in `models/teacher.py`) with three required members: `forward_with_intermediates(inputs)`, `hidden_dim`, `readout_dim`. Use `load_teacher(teacher_config, distance, device, rounds)` to instantiate:
- `type: "mock"` (default, can be omitted) → loads `TeacherWrapper` wrapping a large `StudentDecoder`
- `type: "alphaqubit"` → loads `AlphaQubitAdapter` wrapping `AlphaQubitV2` from `Transformer_0225_fixed.py`

The `AlphaQubitV2` teacher uses X+Z joint stabilizer processing with a fixed RNN+Transformer interleaved architecture (5 RNN + 6 TF layers). Its `forward_with_intermediates()` returns intermediates with `num_t` time steps and `num_z + num_x` stabilizers, which differ from the student's `rounds` and `n_stab` dimensions — so CNN/RNN feature KD is incompatible (`gamma_cnn=0, gamma_rnn=0`). Response KD and readout feature KD work normally.

External teacher config format in YAML:
```yaml
teacher:
  type: "alphaqubit"
  checkpoint: "path/to/model.pt"
  hidden_dim: 256        # d_model
  readout_dim: 256       # equals d_model for AlphaQubitV2
  # n_heads: 8           # optional, default 8
```

### Model Sizes (create_student factory)
| Size   | hidden_dim | conv_dim | readout_dim | CNN blocks |
|--------|-----------|----------|-------------|------------|
| tiny   | 16        | 8        | 8           | 1          |
| small  | 32        | 16       | 16          | 1          |
| medium | 64        | 32       | 32          | 2          |
| large  | 128       | 64       | 64          | 2          |

### Distillation Hook Points
The student exposes intermediates via `return_intermediates=True`:
- **Hook A** (`cnn_features`): `[batch, rounds, n_stab, hidden_dim]` — spatial features after CNN
- **Hook B** (`decoder_states`): `[batch, rounds, n_stab, hidden_dim]` — temporal state after RNN (before CNN)
- **Hook C/D** (`readout_logits`): `[batch, 1]` — output logits

### Distillation Loss Weights (in config YAML)
`total = alpha * L_task + beta * L_response + gamma_cnn * L_cnn_feature + gamma_rnn * L_rnn_feature + gamma_readout * L_readout_feature`

- `alpha`: ground truth CE weight
- `beta`: soft logit KD weight (ResponseKDLoss with temperature scaling)
- `gamma_cnn`: CNN feature alignment weight (FeatureKDLoss with optional projection head)
- `gamma_rnn`: RNN state alignment weight
- `gamma_readout`: readout feature alignment weight (uses readout_dim, not hidden_dim)

When student and teacher hidden dims differ, `FeatureKDLoss` automatically adds a `nn.Linear` projector. The projector parameters are included in the optimizer alongside student parameters.

### Data Generation
Uses the `stim` library to generate rotated surface code memory experiments with SI1000 noise model. Data is generated in-memory (not cached to disk). Key params: `distance` (code distance d), `rounds` (EC rounds), `noise_strength` (physical error rate p), `snr` (soft readout signal-to-noise).

**Online vs Offline mode** (`data.online` in config YAML):
- **Offline** (`online: false`, default): Data generated once at init, reused every epoch. Good for quick tests.
- **Online** (`online: true`): Training data re-sampled every epoch via `set_epoch(epoch)`, using `seed = base_seed + epoch`. Prevents overfitting to a fixed training set. Validation set always stays offline for fair evaluation.

The `stim_generator.py` exposes a two-step API for online mode:
1. `prepare_surface_code_context(distance, rounds, noise_strength)` — one-time circuit compilation + canonical permutations
2. `sample_from_context(context, num_shots, snr, use_soft, seed)` — lightweight per-epoch sampling

The legacy `generate_surface_code_data()` delegates to both and remains backward-compatible.

**Raw detectors for external teachers**: When `include_raw_detectors=True` is passed to `create_dataloaders()` (auto-enabled for `type: "alphaqubit"`), the dataset also stores `raw_detectors: [N, num_detectors]` — the flat, unpermuted detection events that `AlphaQubitV2` expects. The `AlphaQubitAdapter.forward_with_intermediates()` reads `inputs["raw_detectors"]` directly.

**Seed strategy**: Teacher uses `seed=0` → online seeds `{1, 2, ...}`. Student uses `seed=42` → online seeds `{43, 44, ...}`. Validation set uses `seed + 10000` (always offline). No overlap between teacher/student/val seeds.

### Primary Metric
Logical Error Rate (LER) = 1 - accuracy. Per-round LER epsilon derived via: `epsilon = 0.5 * (1 - (1 - 2*E(n))^(1/n))`.

## Config Structure

All configs are YAML in `configs/`. Sections: `data`, `model`, `training`, `logging`, and optionally `teacher` + `distillation` for KD. Checkpoints save to `checkpoints/<name>/` with `best_model.pt`, `final_model.pt`, `config.yaml`, `history.json`.

## Code Layout

- `train.py` — scratch training (no teacher)
- `train_distill.py` — KD training (loads frozen teacher, uses DistillationLoss)
- `models/student.py` — StudentDecoder and `create_student` factory
- `models/teacher.py` — TeacherAdapter ABC, TeacherWrapper, AlphaQubitAdapter, `load_teacher()` unified entry
- `models/Transformer_0225_fixed.py` — AlphaQubitV2 (X+Z joint, RNN+TF interleaved) + FullMapper
- `models/Transformer_0223.py` — AlphaQubitStrictZ (Z-only, legacy) + StrictZMapper
- `models/modules/` — embedder, cnn_block, recurrent, readout
- `distillation/losses.py` — ResponseKDLoss, FeatureKDLoss, DistillationLoss
- `data/` — dataset.py (PyTorch Dataset/DataLoader), stim_generator.py (Stim circuit sampling)
- `evaluation/metrics.py` — LER, accuracy, per-round LER, error suppression ratio
- `run_pipeline.py` — end-to-end pipeline with mock/external teacher modes
- `run_ablations.py` — ablation experiment runner (Group A: signal ablation, Group B: fusion ablation)
- `docs/config_guide.md` — comprehensive YAML config reference
- `docs/teacher_adapter_guide.md` — Teacher adapter integration guide (Chinese)
- `plan.md` — full distillation roadmap (Tasks 1-5, ablation design)
