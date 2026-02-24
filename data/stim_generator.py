"""
基于 Stim 的 rotated surface code memory 实验数据生成。

为训练神经网络解码器生成检测事件和 soft readout 值。
使用 AlphaQubit 论文中的 SI1000 电路去极化噪声模型。
"""

import numpy as np
import stim


def make_surface_code_circuit(
    distance: int,
    rounds: int,
    noise_strength: float = 0.001,
) -> stim.Circuit:
    """
    使用 Stim 内置电路生成器和 SI1000 噪声模型，
    生成 rotated surface code memory 实验电路。

    Args:
        distance: 码距 (d)。该码使用 d^2 个数据量子比特
                  和 d^2-1 个 stabilizer。
        rounds: 纠错轮数。
        noise_strength: SI1000 模型的物理错误率参数 p。

    Returns:
        memory 实验的 stim.Circuit。
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise_strength,
        after_reset_flip_probability=noise_strength,
        before_measure_flip_probability=noise_strength,
        before_round_data_depolarization=noise_strength,
    )
    return circuit


def sample_detection_events(
    circuit: stim.Circuit,
    num_shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从电路中采样检测事件和逻辑可观测量。

    Args:
        circuit: Stim 电路。
        num_shots: 采样次数。

    Returns:
        detection_events: bool 数组 [num_shots, num_detectors]
        logical_observables: bool 数组 [num_shots, num_observables]
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, logical_observables = sampler.sample(
        shots=num_shots, separate_observables=True
    )
    return detection_events.astype(np.float32), logical_observables.astype(np.float32)


def prepare_surface_code_context(
    distance: int,
    rounds: int,
    noise_strength: float = 0.001,
) -> dict:
    """
    一次性预计算 Stim 电路和标准排列（开销较大的部分）。

    在 online 数据生成模式下，此函数只在初始化时调用一次，
    之后每个 epoch 调用 sample_from_context() 即可。

    Args:
        distance: 码距 d。
        rounds: 纠错轮数。
        noise_strength: SI1000 噪声模型的物理错误率。

    Returns:
        context 字典，包含：
            'circuit': stim.Circuit
            'perms': 逐轮标准排列列表
            'n_stabilizers': int
            'distance': int
            'rounds': int
    """
    n_stabilizers = distance * distance - 1
    circuit = make_surface_code_circuit(distance, rounds, noise_strength)
    perms = _compute_canonical_permutations(circuit, rounds, n_stabilizers)
    return {
        "circuit": circuit,
        "perms": perms,
        "n_stabilizers": n_stabilizers,
        "distance": distance,
        "rounds": rounds,
    }


def sample_from_context(
    context: dict,
    num_shots: int,
    snr: float = 10.0,
    use_soft: bool = True,
    seed: int = 42,
) -> dict:
    """
    基于预计算的 context 采样数据（轻量级，每个 epoch 调用）。

    Args:
        context: 由 prepare_surface_code_context() 返回的字典。
        num_shots: 生成样本数。
        snr: soft readout 模拟的信噪比。
        use_soft: 是否生成 soft readout 值。
        seed: 可复现性的随机种子。

    Returns:
        包含以下键的字典：
            'detection_events': float32 [num_shots, rounds, n_stabilizers]
            'soft_events': float32 [num_shots, rounds, n_stabilizers]（如果 use_soft）
            'logical_observables': float32 [num_shots]
            'distance': int
            'rounds': int
            'n_stabilizers': int
    """
    rng = np.random.default_rng(seed)
    circuit = context["circuit"]
    perms = context["perms"]
    n_stabilizers = context["n_stabilizers"]
    distance = context["distance"]
    rounds = context["rounds"]

    sampler = circuit.compile_detector_sampler(seed=seed)
    raw_events, logical_obs = sampler.sample(
        shots=num_shots, separate_observables=True
    )

    num_detectors = raw_events.shape[1]
    expected_detectors = rounds * n_stabilizers
    assert num_detectors == expected_detectors, (
        f"Detector count mismatch: got {num_detectors}, "
        f"expected {expected_detectors} (d={distance}, r={rounds})"
    )

    detection_events = raw_events.reshape(num_shots, rounds, n_stabilizers).astype(
        np.float32
    )

    for r in range(rounds):
        detection_events[:, r, :] = detection_events[:, r, perms[r]]

    logical_observables = logical_obs.astype(np.float32).squeeze(-1)

    result = {
        "detection_events": detection_events,
        "logical_observables": logical_observables,
        "distance": distance,
        "rounds": rounds,
        "n_stabilizers": n_stabilizers,
    }

    if use_soft:
        soft_events = _generate_soft_events(detection_events, snr, rng)
        result["soft_events"] = soft_events

    return result


def generate_surface_code_data(
    distance: int,
    rounds: int,
    num_shots: int,
    noise_strength: float = 0.001,
    snr: float = 10.0,
    use_soft: bool = True,
    seed: int = 42,
) -> dict:
    """
    生成带有可选 soft readout 的 surface code 解码数据。

    这是主要的数据生成函数，向后兼容的便捷入口。
    内部委托给 prepare_surface_code_context() + sample_from_context()。

    Args:
        distance: 码距 d。
        rounds: 纠错轮数。
        num_shots: 生成样本数。
        noise_strength: SI1000 噪声模型的物理错误率。
        snr: soft readout 模拟的信噪比。
        use_soft: 是否生成 soft readout 值。
        seed: 可复现性的随机种子。

    Returns:
        包含以下键的字典：
            'detection_events': float32 [num_shots, rounds, n_stabilizers]
                重塑为逐轮、逐 stabilizer 的二值检测事件。
            'soft_events': float32 [num_shots, rounds, n_stabilizers]（如果 use_soft）
                每个检测事件的 soft 后验概率。
            'logical_observables': float32 [num_shots]
                真实逻辑可观测量结果。
            'distance': int
            'rounds': int
            'n_stabilizers': int
    """
    context = prepare_surface_code_context(distance, rounds, noise_strength)
    return sample_from_context(context, num_shots, snr=snr, use_soft=use_soft, seed=seed)


def _compute_canonical_permutations(circuit, rounds, n_stabilizers):
    """
    计算逐轮排列，将检测器排序对齐到标准（中间轮）排序。

    Stim 的边界轮（首轮和末轮）可能在与中间轮不同的物理位置
    有检测器。每轮的检测器跨越轮内的两个时间步
    （如 X-type 和 Z-type stabilizer）。我们使用 (x, y, t_offset)
    作为匹配键，正确区分位于相同 (x,y) 但不同子轮时间步的检测器。

    对于无法匹配的检测器（边界特有位置），
    按空间最近邻分配到剩余的标准索引。

    Returns:
        np.ndarray 的列表，每轮一个。perms[r][canonical_idx] = stim_idx，
        使得 detection_events[:, r, :] = raw[:, r, perms[r]] 给出标准排序。
    """
    coords = circuit.get_detector_coordinates()
    mid_round = rounds // 2

    def get_base_t(r):
        """获取第 r 轮检测器的最小时间坐标。"""
        times = set()
        for i in range(n_stabilizers):
            det_idx = r * n_stabilizers + i
            times.add(coords[det_idx][2])
        return min(times)

    mid_base_t = get_base_t(mid_round)

    # 标准排序：(x, y, t_offset) -> 索引，使用中间轮
    canon_key = {}
    for i in range(n_stabilizers):
        det_idx = mid_round * n_stabilizers + i
        x, y, t = coords[det_idx][0], coords[det_idx][1], coords[det_idx][2]
        t_offset = t - mid_base_t
        canon_key[(x, y, t_offset)] = i

    perms = []
    for r in range(rounds):
        base_t = get_base_t(r)
        inv_perm = np.full(n_stabilizers, -1, dtype=np.int64)
        used_canonical = set()
        used_stim = set()

        # 第一遍：按 (x, y, t_offset) 匹配
        for i in range(n_stabilizers):
            det_idx = r * n_stabilizers + i
            x, y, t = coords[det_idx][0], coords[det_idx][1], coords[det_idx][2]
            t_offset = t - base_t
            key = (x, y, t_offset)
            if key in canon_key:
                canonical_idx = canon_key[key]
                if canonical_idx not in used_canonical:
                    inv_perm[canonical_idx] = i
                    used_canonical.add(canonical_idx)
                    used_stim.add(i)

        # 第二遍：对未匹配的边界检测器做最近邻回退。
        # 边界轮可能有不在标准集合中的位置的检测器。
        # 将每个映射到空间上最近的未匹配标准位置。
        remaining_canonical = sorted(set(range(n_stabilizers)) - used_canonical)
        remaining_stim = sorted(set(range(n_stabilizers)) - used_stim)

        if remaining_canonical:
            # 获取剩余标准位置的 (x,y)（来自中间轮）
            canon_xy = []
            for c in remaining_canonical:
                det_mid = mid_round * n_stabilizers + c
                canon_xy.append((coords[det_mid][0], coords[det_mid][1]))

            # 贪心最近邻分配
            assigned_canonical = set()
            for s in remaining_stim:
                det_idx = r * n_stabilizers + s
                sx, sy = coords[det_idx][0], coords[det_idx][1]
                best_c = None
                best_dist = float('inf')
                for ci, c in enumerate(remaining_canonical):
                    if c in assigned_canonical:
                        continue
                    cx, cy = canon_xy[ci]
                    dist = (sx - cx) ** 2 + (sy - cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c
                inv_perm[best_c] = s
                assigned_canonical.add(best_c)

        assert len(set(inv_perm)) == n_stabilizers, (
            f"Round {r}: permutation is not a bijection: {inv_perm}"
        )
        perms.append(inv_perm)

    return perms


def _generate_soft_events(
    hard_events: np.ndarray,
    snr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    从硬（二值）检测事件生成 soft 检测事件。

    模拟模拟 I/Q 测量过程：
    - 硬事件 = 0 → "无检测"，模拟信号接近 0
    - 硬事件 = 1 → "有检测"，模拟信号接近 1
    - 基于 SNR 添加高斯噪声
    - 通过贝叶斯规则转换为后验概率 P(event=1 | z)

    安全约束：soft 值被 clamp 确保绝不越过 0.5 边界，
    保持硬逻辑值不变。

    Args:
        hard_events: [num_shots, rounds, n_stabilizers] 二值 {0, 1}
        snr: 信噪比。
        rng: 随机数生成器。

    Returns:
        soft_events: [num_shots, rounds, n_stabilizers]，值域 (0, 1)
            hard=0 时值在 (eps, 0.5-eps)，hard=1 时值在 (0.5+eps, 1-eps)。
    """
    shape = hard_events.shape
    sigma = 1.0 / np.sqrt(snr)

    # 模拟模拟测量：z = 真实值 + 噪声
    noise = rng.normal(0, sigma, size=shape).astype(np.float32)
    z = hard_events + noise

    # 通过等先验的贝叶斯规则计算后验概率：
    # P(1|z) = P(z|1) / (P(z|0) + P(z|1))
    # log P(z|k) = -SNR/2 * (z-k)^2
    log_p0 = -0.5 * snr * z**2
    log_p1 = -0.5 * snr * (z - 1.0) ** 2

    # 数值稳定的计算
    log_max = np.maximum(log_p0, log_p1)
    p1 = np.exp(log_p1 - log_max) / (
        np.exp(log_p0 - log_max) + np.exp(log_p1 - log_max)
    )

    # 安全 clamp：soft 值不可越过 0.5 边界
    eps = 1e-4
    mask_0 = hard_events < 0.5
    mask_1 = hard_events >= 0.5
    p1[mask_0] = np.clip(p1[mask_0], eps, 0.5 - eps)
    p1[mask_1] = np.clip(p1[mask_1], 0.5 + eps, 1.0 - eps)

    return p1.astype(np.float32)
