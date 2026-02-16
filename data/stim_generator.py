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
    sampler = circuit.compile_detector_sampler()
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
