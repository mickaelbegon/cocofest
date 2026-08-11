import json

import numpy as np
import pytest

from examples.fes_multibody.cycling.build_acados_control_seed import (
    build_hybrid_control_seed,
)


MUSCLES = ("Biceps", "Triceps", "Delt_ant", "Delt_post")


def _metadata(cycles):
    return {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "model_formulation": "periodic_node",
        "mechanical_formulation": "reduced",
        "cycles_per_window": cycles,
        "stimulations_per_cycle": 3,
        "constant_crank_torque": -0.0,
        "torque_application": "constant",
        "calcium_forcing_formulation": "exact_exponential_periodic_node",
        "ding_sum_stim_truncation": 6,
        "activate_force_length_relationship": True,
        "activate_force_velocity_relationship": True,
        "activate_passive_force_relationship": True,
        "pulse_width_minimum_policy": "model_pd0",
        "pulse_width_maximum_s": 0.0006,
        "producer_solver": "acados" if cycles > 1 else "ipopt",
        "producer_transcription_profile": "acados-irk" if cycles > 1 else "radau5",
    }


def _write_seed(path, *, cycles, control_offset=0.0):
    payload = {
        "states__theta": np.linspace(-2 * np.pi, -4 * np.pi, 13)[None, :],
        "states__omega": np.full((1, 13), -2 * np.pi),
    }
    for index, muscle in enumerate(MUSCLES):
        controls = (
            140e-6
            + control_offset
            + index * 2e-6
            + np.arange(cycles * 3, dtype=float)[None, :] * 1e-6
        )
        payload[f"controls__last_pulse_width_{muscle}"] = controls
    payload["metadata__json"] = np.asarray(
        json.dumps(_metadata(cycles), sort_keys=True, separators=(",", ":"))
    )
    np.savez(path, **payload)


def test_hybrid_control_seed_preserves_states_and_selects_one_cycle(tmp_path):
    common_path = tmp_path / "common.npz"
    acados_path = tmp_path / "acados.npz"
    output_path = tmp_path / "hybrid.npz"
    _write_seed(common_path, cycles=1)
    _write_seed(acados_path, cycles=4, control_offset=10e-6)

    summary = build_hybrid_control_seed(common_path, acados_path, output_path, cycle=3)

    with np.load(common_path, allow_pickle=False) as common, np.load(
        acados_path, allow_pickle=False
    ) as acados, np.load(output_path, allow_pickle=False) as hybrid:
        np.testing.assert_array_equal(hybrid["states__theta"], common["states__theta"])
        np.testing.assert_array_equal(hybrid["states__omega"], common["states__omega"])
        for muscle in MUSCLES:
            key = f"controls__last_pulse_width_{muscle}"
            np.testing.assert_array_equal(hybrid[key], acados[key][:, 6:9])
        metadata = json.loads(str(hybrid["metadata__json"].item()))

    assert summary["cycle"] == 3
    assert summary["state_count"] == 2
    assert summary["control_count"] == 4
    assert metadata["initial_control_seed"]["source_cycle"] == 3
    assert metadata["initial_control_seed"]["source_cycles"] == 4


def test_hybrid_control_seed_rejects_an_out_of_range_cycle(tmp_path):
    common_path = tmp_path / "common.npz"
    acados_path = tmp_path / "acados.npz"
    _write_seed(common_path, cycles=1)
    _write_seed(acados_path, cycles=2)

    with pytest.raises(ValueError, match="available range is 1..2"):
        build_hybrid_control_seed(
            common_path, acados_path, tmp_path / "hybrid.npz", cycle=3
        )


def test_hybrid_control_seed_rejects_a_physical_model_mismatch(tmp_path):
    common_path = tmp_path / "common.npz"
    acados_path = tmp_path / "acados.npz"
    _write_seed(common_path, cycles=1)
    _write_seed(acados_path, cycles=2)
    with np.load(acados_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    metadata = json.loads(str(payload["metadata__json"].item()))
    metadata["activate_passive_force_relationship"] = False
    payload["metadata__json"] = np.asarray(json.dumps(metadata))
    np.savez(acados_path, **payload)

    with pytest.raises(ValueError, match="activate_passive_force_relationship"):
        build_hybrid_control_seed(
            common_path, acados_path, tmp_path / "hybrid.npz", cycle=1
        )
