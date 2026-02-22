"""
Stim-based data generation for rotated surface code memory experiments.

Generates detection events and soft readout values for training
neural network decoders. Follows the SI1000 circuit depolarizing
noise model used in the AlphaQubit paper.
"""

import numpy as np
import stim


def make_surface_code_circuit(
    distance: int,
    rounds: int,
    noise_strength: float = 0.001,
) -> stim.Circuit:
    """
    Generate a rotated surface code memory experiment circuit using Stim's
    built-in circuit generation with SI1000 noise model.

    Args:
        distance: Code distance (d). The code uses d^2 data qubits
                  and d^2-1 stabilizers.
        rounds: Number of error-correction rounds.
        noise_strength: Physical error rate parameter p for the SI1000 model.

    Returns:
        A stim.Circuit for the memory experiment.
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
    Sample detection events and logical observables from the circuit.

    Args:
        circuit: The Stim circuit.
        num_shots: Number of shots to sample.

    Returns:
        detection_events: bool array [num_shots, num_detectors]
        logical_observables: bool array [num_shots, num_observables]
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, logical_observables = sampler.sample(
        shots=num_shots, separate_observables=True
    )
    return detection_events.astype(np.float32), logical_observables.astype(np.float32)


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
    Generate surface code decoding data with optional soft readout.

    This is the main data generation function. It:
    1. Creates a Stim circuit for a rotated surface code memory experiment
    2. Samples detection events and logical observables
    3. Reshapes detection events to [rounds, n_stabilizers] per sample
    4. Optionally generates soft detection events with simulated I/Q noise

    For soft readout, we simulate measurement uncertainty by:
    - Treating each hard detection event as a noisy binary observation
    - Generating a posterior probability from a noisy "analog" signal
    - SAFETY: soft values NEVER cross the 0.5 boundary

    Args:
        distance: Code distance d.
        rounds: Number of error-correction rounds.
        num_shots: Number of samples to generate.
        noise_strength: Physical error rate for SI1000 noise model.
        snr: Signal-to-noise ratio for soft readout simulation.
        use_soft: Whether to generate soft readout values.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
            'detection_events': float32 [num_shots, rounds, n_stabilizers]
                Binary detection events reshaped to per-round, per-stabilizer.
            'soft_events': float32 [num_shots, rounds, n_stabilizers] (if use_soft)
                Soft posterior probabilities for each detection event.
            'logical_observables': float32 [num_shots]
                Ground truth logical observable outcomes.
            'distance': int
            'rounds': int
            'n_stabilizers': int
    """
    rng = np.random.default_rng(seed)

    n_stabilizers = distance * distance - 1

    circuit = make_surface_code_circuit(distance, rounds, noise_strength)

    # Stim detection events are already correctly computed
    # They represent temporal differences of stabilizer measurements
    sampler = circuit.compile_detector_sampler(seed=seed)
    raw_events, logical_obs = sampler.sample(
        shots=num_shots, separate_observables=True
    )

    # raw_events shape: [num_shots, num_detectors]
    # For rotated surface code with `rounds` rounds:
    #   num_detectors = rounds * n_stabilizers
    # (Stim computes boundary detectors correctly)
    num_detectors = raw_events.shape[1]
    expected_detectors = rounds * n_stabilizers
    assert num_detectors == expected_detectors, (
        f"Detector count mismatch: got {num_detectors}, "
        f"expected {expected_detectors} (d={distance}, r={rounds})"
    )

    # Reshape to [num_shots, rounds, n_stabilizers]
    detection_events = raw_events.reshape(num_shots, rounds, n_stabilizers).astype(
        np.float32
    )

    # Reorder each round's detectors to canonical (middle-round) ordering.
    # Stim's boundary rounds may order detectors differently from middle rounds;
    # this ensures index i always corresponds to the same physical stabilizer.
    perms = _compute_canonical_permutations(circuit, rounds, n_stabilizers)
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


def _compute_canonical_permutations(circuit, rounds, n_stabilizers):
    """
    Compute per-round permutations to align detector ordering to canonical (middle round).

    Stim's boundary rounds (first and last) may have detectors at different physical
    positions than middle rounds. Each round's detectors span two time steps within
    the round (e.g., X-type and Z-type stabilizers). We use (x, y, t_offset) as the
    matching key to correctly disambiguate detectors at the same (x,y) but different
    sub-round time steps.

    For detectors that cannot be matched (boundary-specific positions), they are
    assigned to remaining canonical indices in sorted order.

    Returns:
        List of np.ndarray, one per round. perms[r][canonical_idx] = stim_idx,
        so detection_events[:, r, :] = raw[:, r, perms[r]] gives canonical ordering.
    """
    coords = circuit.get_detector_coordinates()
    mid_round = rounds // 2

    def get_base_t(r):
        """Get the minimum time coordinate for detectors in round r."""
        times = set()
        for i in range(n_stabilizers):
            det_idx = r * n_stabilizers + i
            times.add(coords[det_idx][2])
        return min(times)

    mid_base_t = get_base_t(mid_round)

    # Canonical ordering: (x, y, t_offset) -> index, using middle round
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

        # First pass: match by (x, y, t_offset)
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

        # Second pass: nearest-neighbor fallback for unmatched boundary detectors.
        # Boundary rounds may have detectors at positions not in the canonical set.
        # Map each to the spatially closest unmatched canonical position.
        remaining_canonical = sorted(set(range(n_stabilizers)) - used_canonical)
        remaining_stim = sorted(set(range(n_stabilizers)) - used_stim)

        if remaining_canonical:
            # Get (x,y) for remaining canonical positions (from middle round)
            canon_xy = []
            for c in remaining_canonical:
                det_mid = mid_round * n_stabilizers + c
                canon_xy.append((coords[det_mid][0], coords[det_mid][1]))

            # Greedy nearest-neighbor assignment
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
    Generate soft detection events from hard (binary) detection events.

    Simulates the analog I/Q measurement process:
    - Hard event = 0 → "no detection", analog signal near 0
    - Hard event = 1 → "detection", analog signal near 1
    - Gaussian noise added based on SNR
    - Convert to posterior probability P(event=1 | z) via Bayes' rule

    SAFETY CONSTRAINT: soft values are clamped to never cross the 0.5
    boundary, preserving the hard logical value.

    Args:
        hard_events: [num_shots, rounds, n_stabilizers] binary {0, 1}
        snr: Signal-to-noise ratio.
        rng: Random number generator.

    Returns:
        soft_events: [num_shots, rounds, n_stabilizers] in (0, 1)
            Values in (eps, 0.5-eps) for hard=0, (0.5+eps, 1-eps) for hard=1.
    """
    shape = hard_events.shape
    sigma = 1.0 / np.sqrt(snr)

    # Simulate analog measurement: z = true_value + noise
    noise = rng.normal(0, sigma, size=shape).astype(np.float32)
    z = hard_events + noise

    # Posterior probability via Bayes' rule with equal priors:
    # P(1|z) = P(z|1) / (P(z|0) + P(z|1))
    # log P(z|k) = -SNR/2 * (z-k)^2
    log_p0 = -0.5 * snr * z**2
    log_p1 = -0.5 * snr * (z - 1.0) ** 2

    # Numerically stable computation
    log_max = np.maximum(log_p0, log_p1)
    p1 = np.exp(log_p1 - log_max) / (
        np.exp(log_p0 - log_max) + np.exp(log_p1 - log_max)
    )

    # Safety clamp: soft values must NOT cross 0.5 boundary
    eps = 1e-4
    mask_0 = hard_events < 0.5
    mask_1 = hard_events >= 0.5
    p1[mask_0] = np.clip(p1[mask_0], eps, 0.5 - eps)
    p1[mask_1] = np.clip(p1[mask_1], 0.5 + eps, 1.0 - eps)

    return p1.astype(np.float32)
