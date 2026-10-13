"""E8 parameter single source of truth (spec: docs/E8_Design_Spec_2026-10-11.md).

E8 adds the decision rules of published approaches that E4 did not cover, on
the same seed-identical E3 ensembles and the same fault set as E4:

  M   mean-tracking time-aware threshold (time-aware readout family):
      |z - mu_model(t)| > theta0; identical to the constant tolerance on the
      differential pair, whose model mean is zero, and evaluated on the bias
      mapping (encoding a) where the mean drifts;
  O   online variance-adaptive threshold (adaptive-threshold checker and
      variance-based ABFT family): |z_n| > k * sqrt(v_{n-1}), with v an
      exponentially weighted mean of z^2 updated every inference, either per
      checksum group (O_g) or pooled over the K groups of an array (O_p),
      alpha = 2^-s for s in ewma_shifts; k is calibrated at t0 for FPR0.

Physics comes from E1Config and ensemble seed coordinates from E3Config, so the
E4 detectors recomputed here reproduce the E4 official record (spec gate G1).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["E8Config"]


@dataclass(frozen=True)
class E8Config:
    # comparison slice: identical to E4 (encoding b, m = 16, c = 1)
    sigma_nu_on: tuple = (0.005, 0.01, 0.02, 0.04)  # ratio readout on
    sigma_off_slice: float = 0.02                   # ratio readout off
    m: int = 16
    c: int = 1
    adc_bits: tuple = (6, 8, 10)                    # E4 exact-equality family

    # online variance-adaptive threshold
    ewma_shifts: tuple = (6, 10)   # alpha = 2^-s, time constants 64 and 1024 inferences
    ewma_passes: int = 3           # circular passes over the stationary stream;
                                   # alarms are evaluated on the last pass only

    # mean-tracking slice on the bias mapping (encoding a, ratio readout on)
    enc_a_m: int = 16
    enc_a_c: int = 1

    # discrimination probes: E4 seed branch and draw order, so the fault set
    # is the one E4 used (gate G2 checks the single-inference p1 bit for bit)
    probe_deltas: tuple = (0.3, 0.95)
    probe_t: tuple = (1.0, 100.0, 10000.0)
    n_trials: int = 4000
    fault_seed_branch: int = 8
    trial_chunk: int = 400

    fpr0: float = 0.01
