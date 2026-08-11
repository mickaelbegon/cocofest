from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "run_full_horizon_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_full_horizon_benchmark", SCRIPT_PATH)
full_horizon = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = full_horizon
SPEC.loader.exec_module(full_horizon)


def test_rho_initial_state_homotopy_minimum_step_is_finer_than_one_over_32():
    assert full_horizon.RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP == pytest.approx(1 / 256)
    assert full_horizon.RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP < 1 / 32


@pytest.mark.parametrize(
    ("maximum", "expected_tail"),
    (
        (2, [2]),
        (3, [2, 3]),
        (32, [30, 31, 32]),
        (60, [58, 59, 60]),
        (100, [98, 99, 100]),
    ),
)
def test_horizon_sweep_targets_grow_one_cycle_at_a_time(maximum, expected_tail):
    targets = full_horizon.horizon_sweep_targets(maximum)

    assert targets == sorted(set(targets))
    assert targets[-1] == maximum
    assert targets[-len(expected_tail) :] == expected_tail


def test_horizon_sweep_requires_the_two_rho_bootstrap_cycles():
    with pytest.raises(ValueError, match="at least two"):
        full_horizon.horizon_sweep_targets(1)


def test_adaptive_target_uses_three_cycles_and_clips_the_tail():
    assert full_horizon.adaptive_continuation_target(2, 20, 3) == 5
    assert full_horizon.adaptive_continuation_target(20, 22, 3) == 22


def test_jump_objective_gate_rejects_excessive_degradation():
    accepted = full_horizon.objective_gate(100.4, 100.0, 0.005)
    rejected = full_horizon.objective_gate(100.6, 100.0, 0.005)

    assert accepted["passes"]
    assert accepted["relative_degradation"] == pytest.approx(0.004)
    assert not rejected["passes"]


def test_jump_objective_gate_marks_missing_measurement_incomparable():
    gate = full_horizon.objective_gate(None, 100.0, 0.005)

    assert not gate["comparable"]


def test_adaptive_continuation_falls_back_to_one_then_retries_jump(
    tmp_path, monkeypatch
):
    objectives = tmp_path / "objectives"
    objectives.mkdir()

    def result(name, objective):
        path = objectives / f"{name}.json"
        path.write_text(
            json.dumps({"results": [{"window_objective_sum": objective}]}),
            encoding="utf-8",
        )
        return path

    bootstrap_solution = tmp_path / "fho-2.npz"
    bootstrap_solution.touch()
    bootstrap_result = result("fho-2", 2.0)
    report = {
        "full_horizon_attempts": [
            {
                "cycles": 2,
                "success": True,
                "accepted_for_continuation": True,
                "result_path": str(bootstrap_result),
                "solution_path": str(bootstrap_solution),
            }
        ],
        "extension_rho_attempts": [],
        "adaptive_fallback_events": [],
        "largest_successful_cycles": 2,
        "homotopy_constructed_cycles": 2,
    }
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        continuation_step_cycles=3,
        jump_objective_relative_tolerance=0.005,
        max_cycles=5,
    )
    extension_number = 0

    def fake_extension(call_args, **kwargs):
        nonlocal extension_number
        extension_number += 1
        solution = tmp_path / f"rho-{extension_number}.npz"
        solution.touch()
        return {
            "success": True,
            "infrastructure_error": False,
            "failure_kind": None,
            "solution_path": str(solution),
            "result_path": str(result(f"rho-{extension_number}", 1.0)),
        }

    horizon_targets = []

    def fake_horizon(call_args, **kwargs):
        cycles = kwargs["cycles"]
        horizon_targets.append(cycles)
        objective = 10.0 if horizon_targets == [5] else float(cycles)
        solution = tmp_path / f"fho-{cycles}-{len(horizon_targets)}.npz"
        solution.touch()
        return {
            "cycles": cycles,
            "success": True,
            "infrastructure_error": False,
            "failure_kind": None,
            "solution_path": str(solution),
            "result_path": str(
                result(f"fho-{cycles}-{len(horizon_targets)}", objective)
            ),
        }

    monkeypatch.setattr(full_horizon, "_run_extension_rho", fake_extension)
    monkeypatch.setattr(full_horizon, "_run_horizon_attempt", fake_horizon)
    monkeypatch.setattr(
        full_horizon,
        "append_rho_extension_cycle",
        lambda source, extension, output: (
            output.parent.mkdir(parents=True, exist_ok=True), output.touch()
        ),
    )
    monkeypatch.setattr(full_horizon, "_write_report", lambda *args: None)
    monkeypatch.setattr(full_horizon, "_write_markdown", lambda *args: None)

    return_code = full_horizon._continue_adaptively(
        args,
        report=report,
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        rho_seed_path=tmp_path / "rho-reference.npz",
        effective_max_cycles=5,
        current_cycles=2,
        current_full_solution=bootstrap_solution,
        rss_limit_bytes=1024,
    )

    assert return_code == 0
    assert horizon_targets == [5, 3, 5]
    assert report["largest_successful_cycles"] == 5
    assert report["adaptive_fallback_events"] == [
        {
            "from_cycles": 2,
            "rejected_target_cycles": 5,
            "rejected_step_cycles": 3,
            "reason": "jump_objective_degradation",
            "fallback_step_cycles": 1,
        }
    ]


def test_refinement_fills_only_the_last_coarse_interval():
    assert full_horizon.refinement_targets(60, 70) == list(range(61, 70))
    assert full_horizon.refinement_targets(12, 13) == []


def test_automatic_rss_limits_match_the_two_ci_machine_classes():
    assert full_horizon.automatic_rss_limit_gib(16 * full_horizon.GIB) == 12.5
    assert full_horizon.automatic_rss_limit_gib(128 * full_horizon.GIB) == 97.5


def test_rho_seed_prefix_preserves_complete_cycle_layout_and_metadata(
    tmp_path,
):
    source = tmp_path / "rho.npz"
    output = tmp_path / "prefix.npz"
    metadata = {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "cycles_per_window": 4,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        source,
        states__theta=np.arange(17, dtype=float).reshape(1, 17),
        states__A_Biceps=np.arange(34, dtype=float).reshape(2, 17),
        controls__last_pulse_width_Biceps=np.arange(16, dtype=float).reshape(1, 16),
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    written_metadata = full_horizon.write_rho_seed_prefix(source, output, 3)

    with np.load(output, allow_pickle=False) as data:
        assert data["states__theta"].shape == (1, 13)
        assert data["states__A_Biceps"].shape == (2, 13)
        assert data["controls__last_pulse_width_Biceps"].shape == (1, 12)
        persisted_metadata = json.loads(str(data["metadata__json"].item()))

    assert written_metadata == persisted_metadata
    assert persisted_metadata["cycles_per_window"] == 3
    assert persisted_metadata["producer_source_cycles"] == 4
    assert persisted_metadata["producer_mode"] == "receding_horizon_prefix"


def test_rho_seed_prefix_rejects_non_integral_cycle_layout(tmp_path):
    source = tmp_path / "rho.npz"
    np.savez(
        source,
        states__theta=np.zeros((1, 10)),
        controls__last_pulse_width_Biceps=np.zeros((1, 8)),
        metadata__json=np.asarray(json.dumps({"cycles_per_window": 4})),
    )

    with pytest.raises(ValueError, match="State seed"):
        full_horizon.write_rho_seed_prefix(source, tmp_path / "prefix.npz", 2)


def test_rho_seed_cycle_extracts_one_based_cycle(tmp_path):
    source = tmp_path / "rho.npz"
    output = tmp_path / "cycle-2.npz"
    np.savez(
        source,
        states__theta=np.arange(13, dtype=float).reshape(1, 13),
        controls__pulse=np.arange(12, dtype=float).reshape(1, 12),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 3, "mechanical_formulation": "reduced"})
        ),
    )

    metadata = full_horizon.write_rho_seed_cycle(source, output, 2)

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["states__theta"], [[4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(data["controls__pulse"], [[4, 5, 6, 7]])
    assert metadata["cycles_per_window"] == 1
    assert metadata["producer_cycle_number"] == 2


def test_rho_initial_state_homotopy_hits_requested_fraction(tmp_path):
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    target = tmp_path / "fho-2.npz"
    output = tmp_path / "stage.npz"
    one_cycle_metadata = {
        "cycles_per_window": 1,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        source,
        states__theta=np.asarray([[0.0, -1.0, -2.0]]),
        states__F_Biceps=np.asarray([[10.0, 9.0, 8.0]]),
        controls__pulse=np.asarray([[0.2, 0.3]]),
        metadata__json=np.asarray(json.dumps(one_cycle_metadata)),
    )
    np.savez(
        reference,
        states__theta=np.asarray([[0.0, -1.0, -2.0]]),
        states__F_Biceps=np.asarray([[10.0, 9.0, 8.0]]),
        controls__pulse=np.asarray([[0.2, 0.3]]),
        metadata__json=np.asarray(
            json.dumps({**one_cycle_metadata, "producer_cycle_number": 3})
        ),
    )
    np.savez(
        target,
        states__theta=np.asarray([[0.0, -2.0, -4.0, -6.0, -8.0]]),
        states__F_Biceps=np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0]]),
        controls__pulse=np.asarray([[0.2, 0.3, 0.4, 0.5]]),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 2, "mechanical_formulation": "reduced"})
        ),
    )

    metadata = full_horizon.write_rho_initial_state_homotopy_seed(
        source, reference, target, output, 0.5
    )

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["states__theta"], [[-4.0, -5.0, -6.0]])
        assert data["states__F_Biceps"][0, 0] == pytest.approx(30.0)
        assert data["states__F_Biceps"][0, -1] == pytest.approx(8.0)
        np.testing.assert_array_equal(data["controls__pulse"], [[0.2, 0.3]])
    assert metadata["homotopy_fraction"] == 0.5
    assert metadata["homotopy_reference_cycle_number"] == 3


def test_rho_initial_state_homotopy_aligns_equivalent_theta_winding(tmp_path):
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    target = tmp_path / "fho-3.npz"
    output = tmp_path / "stage.npz"
    metadata = {"cycles_per_window": 1, "mechanical_formulation": "reduced"}
    reference_theta = np.asarray([[-8.0 * np.pi, -9.0 * np.pi, -10.0 * np.pi]])
    np.savez(
        source,
        states__theta=reference_theta,
        metadata__json=np.asarray(json.dumps(metadata)),
    )
    np.savez(
        reference,
        states__theta=reference_theta,
        metadata__json=np.asarray(json.dumps(metadata)),
    )
    np.savez(
        target,
        states__theta=np.asarray([[0.0, -2.0 * np.pi, -6.0 * np.pi]]),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 3, "mechanical_formulation": "reduced"})
        ),
    )

    written = full_horizon.write_rho_initial_state_homotopy_seed(
        source, reference, target, output, 0.25
    )

    with np.load(output, allow_pickle=False) as data:
        theta = data["states__theta"]
    np.testing.assert_allclose(theta[:, 0], [-6.0 * np.pi])
    np.testing.assert_allclose(np.diff(theta), np.diff(reference_theta))
    assert written["homotopy_theta_winding_shift"] == pytest.approx(2.0 * np.pi)


def test_fho_terminal_continuation_starts_exactly_at_fho_terminal_state(tmp_path):
    source = tmp_path / "fho-2.npz"
    output = tmp_path / "next-cycle.npz"
    metadata = {
        "cycles_per_window": 2,
        "mechanical_formulation": "reduced",
    }
    theta = np.asarray([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
    fatigue = np.asarray([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    controls = np.arange(6, dtype=float).reshape(1, 6)
    np.savez(
        source,
        states__theta=theta,
        states__A_Biceps=fatigue,
        controls__last_pulse_width_Biceps=controls,
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    written = full_horizon.write_fho_terminal_continuation_seed(source, output)

    with np.load(output, allow_pickle=False) as data:
        next_theta = data["states__theta"]
        next_fatigue = data["states__A_Biceps"]
        next_controls = data["controls__last_pulse_width_Biceps"]
        persisted = json.loads(str(data["metadata__json"].item()))

    np.testing.assert_allclose(next_theta[:, 0], theta[:, -1])
    np.testing.assert_allclose(next_fatigue[:, 0], fatigue[:, -1])
    assert next_theta[-1, -1] == pytest.approx(-9.0)
    assert next_fatigue[0, -1] == pytest.approx(0.1)
    np.testing.assert_array_equal(next_controls, controls[:, -3:])
    assert written == persisted
    assert persisted["cycles_per_window"] == 1
    assert persisted["producer_source_cycles"] == 2


def test_rho_extension_replaces_the_carrier_boundary_before_appending(tmp_path):
    prefix = tmp_path / "rho-prefix.npz"
    extension = tmp_path / "rho-extension.npz"
    output = tmp_path / "fho-plus-rho.npz"
    prefix_metadata = {
        "cycles_per_window": 2,
        "mechanical_formulation": "reduced",
    }
    extension_metadata = {
        "cycles_per_window": 1,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        prefix,
        states__theta=np.arange(7, dtype=float).reshape(1, 7),
        controls__pulse=np.arange(6, dtype=float).reshape(1, 6),
        metadata__json=np.asarray(json.dumps(prefix_metadata)),
    )
    np.savez(
        extension,
        states__theta=np.asarray([[20.0, 21.0, 22.0, 23.0]]),
        controls__pulse=np.asarray([[30.0, 31.0, 32.0]]),
        metadata__json=np.asarray(json.dumps(extension_metadata)),
    )

    written = full_horizon.append_rho_extension_cycle(prefix, extension, output)

    with np.load(output, allow_pickle=False) as data:
        theta = data["states__theta"]
        pulse = data["controls__pulse"]
        persisted = json.loads(str(data["metadata__json"].item()))

    assert theta.shape == (1, 10)
    assert pulse.shape == (1, 9)
    np.testing.assert_array_equal(theta[0, 6:], [20.0, 21.0, 22.0, 23.0])
    np.testing.assert_array_equal(pulse[0, -3:], [30.0, 31.0, 32.0])
    assert written == persisted
    assert persisted["cycles_per_window"] == 3
    assert persisted["replaced_reduced_boundary_maximum_absolute_change"] == 14.0


def test_benchmark_success_requires_the_complete_physical_horizon(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": True,
                        "solver_success": True,
                        "physical_success": True,
                        "solver": "madnlp",
                        "mode": "single_shot",
                        "covered_cycles": 30,
                        "physically_validated_cycles": 30,
                    }
                ],
                "configurations": {
                    "madnlp": {
                        "single_shot": True,
                        "mechanical_formulation": "full",
                        "cycles_per_window": 30,
                        "n_windows": 30,
                        "use_sx": False,
                        "madnlp_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=30,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=100,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="rho",
        expected_cycles=30,
        expected_solver="madnlp",
    )


def test_validated_cycles_retains_a_shorter_rho_prefix(tmp_path):
    result_path = tmp_path / "rho-result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "solver": "ipopt",
                        "mode": "rho",
                        "success": False,
                        "covered_cycles": 43,
                        "physically_validated_cycles": 42,
                    }
                ],
                "configurations": {
                    "ipopt": {
                        "single_shot": False,
                        "mechanical_formulation": "reduced",
                        "cycles_per_window": 1,
                        "n_windows": 100,
                        "use_sx": True,
                        "ipopt_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        full_horizon._benchmark_validated_cycles(
            result_path,
            expected_mode="rho",
            expected_solver="ipopt",
            expected_requested_cycles=100,
        )
        == 42
    )


def test_rho_extension_accepts_a_valid_cycle_rejected_by_endurance_semantics(
    tmp_path,
):
    result_path = tmp_path / "extension.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": False,
                        "solver_success": True,
                        "physical_success": False,
                        "solver": "ipopt",
                        "mode": "rho",
                        "covered_cycles": 1,
                        "physically_validated_cycles": 1,
                        "windows": [{"validated": True}],
                        "fatigue_endurance_outcome": {
                            "accepted": False,
                            "evidence": ["ding_force_capacity_decreased"],
                        },
                    }
                ],
                "configurations": {
                    "ipopt": {
                        "single_shot": False,
                        "mechanical_formulation": "reduced",
                        "cycles_per_window": 1,
                        "n_windows": 1,
                        "use_sx": True,
                        "ipopt_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert full_horizon._rho_extension_success(result_path)


def test_unknown_mumps_warning_accepts_monitored_string_path(tmp_path):
    log_path = tmp_path / "solver.log"
    log_path.write_text(
        "libMAD WARNING: option linear_solver is of unknown type mumps, ignoring\n",
        encoding="utf-8",
    )

    assert full_horizon._log_has_unknown_mumps_warning(str(log_path))


def test_solver_chances_keep_independent_logs_and_results(tmp_path, monkeypatch):
    observed = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        workspace=tmp_path,
        poll_interval_s=0.5,
        attempt_timeout_s=30.0,
    )
    monkeypatch.setattr(
        full_horizon,
        "write_rho_seed_prefix",
        lambda source, destination, cycles: destination.parent.mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        full_horizon,
        "_full_horizon_command",
        lambda *command_args, **command_kwargs: ["solver"],
    )

    def fake_run_monitored(command, **kwargs):
        observed["log_path"] = kwargs["log_path"]
        return full_horizon.MonitoredRun(
            command=command,
            return_code=1,
            peak_rss_bytes=0,
            elapsed_s=1.0,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        )

    monkeypatch.setattr(full_horizon, "run_monitored", fake_run_monitored)

    attempt = full_horizon._run_horizon_attempt(
        args,
        rho_seed=tmp_path / "rho.npz",
        cycles=2,
        phase="coarse",
        chance=2,
        rss_limit_bytes=1024,
    )

    expected_dir = tmp_path / "output" / "full-horizon-0002" / "chance-2"
    assert observed["log_path"] == expected_dir / "solver.log"
    assert attempt["result_path"] == str(expected_dir / "result.json")
    assert attempt["seed_origin"] == "rho_prefix"


def test_horizon_attempt_passes_the_certified_full_prefix(tmp_path, monkeypatch):
    observed = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        workspace=tmp_path,
        poll_interval_s=0.5,
        attempt_timeout_s=30.0,
    )
    prefix = tmp_path / "certified-full.npz"
    monkeypatch.setattr(
        full_horizon,
        "write_rho_seed_prefix",
        lambda source, destination, cycles: destination.parent.mkdir(
            parents=True, exist_ok=True
        ),
    )

    def fake_command(*command_args, **command_kwargs):
        observed["prefix"] = command_kwargs["prefix_solution_path"]
        return ["solver"]

    monkeypatch.setattr(full_horizon, "_full_horizon_command", fake_command)
    monkeypatch.setattr(
        full_horizon,
        "run_monitored",
        lambda command, **kwargs: full_horizon.MonitoredRun(
            command=command,
            return_code=1,
            peak_rss_bytes=0,
            elapsed_s=1.0,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        ),
    )

    attempt = full_horizon._run_horizon_attempt(
        args,
        rho_seed=tmp_path / "rho.npz",
        cycles=2,
        phase="coarse",
        chance=1,
        rss_limit_bytes=1024,
        prefix_solution_path=prefix,
    )

    assert observed["prefix"] == prefix
    assert attempt["seed_origin"] == "rho_plus_certified_fho_prefix"
    assert attempt["prefix_solution_path"] == str(prefix)


def test_workflow_has_an_isolated_mx_mumps_full_horizon_mode():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    full_job = workflow.split("\n  full-horizon:", maxsplit=1)[1].split(
        "\n  screen-report:", maxsplit=1
    )[0]

    assert "inputs.cycles == 'full_horizon'" in full_job
    assert "--single-shot" in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--madnlp-linear-solver"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"mumps"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--ipopt-no-use-sx"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--optional-nlp-periodic-ipopt-hot-start"' in SCRIPT_PATH.read_text(
        encoding="utf-8"
    )
    assert "--memory-limit-gib" in full_job
    assert "--continuation-step-cycles 3" in full_job
    assert "--jump-objective-relative-tolerance 0.005" in full_job
    assert "full_horizon_max_cycles" in workflow
    assert "cycling-full-horizon-${{ github.run_id }}" in full_job


def test_rho_and_full_horizon_use_the_intended_solver_contract(tmp_path):
    args = SimpleNamespace(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=100,
    )

    rho = full_horizon._rho_command(args, tmp_path / "rho.json", tmp_path / "rho.npz")
    full = full_horizon._full_horizon_command(
        args,
        60,
        tmp_path / "prefix.npz",
        tmp_path / "full.json",
        tmp_path / "full.npz",
        prefix_solution_path=tmp_path / "previous-full.npz",
    )
    one_cycle_full = full_horizon._full_horizon_command(
        args,
        1,
        tmp_path / "one-cycle-prefix.npz",
        tmp_path / "one-cycle-full.json",
        tmp_path / "one-cycle-full.npz",
    )
    paired_reduced = full_horizon._full_horizon_command(
        args,
        2,
        tmp_path / "two-cycle-prefix.npz",
        tmp_path / "two-cycle-reduced.json",
        tmp_path / "two-cycle-reduced.npz",
        mechanical_formulation="reduced",
    )

    assert rho[rho.index("--solvers") + 1] == "ipopt"
    assert full[full.index("--solvers") + 1] == "madnlp"
    assert "--single-shot" not in rho
    assert "--allow-partial-receding-horizon-solution-output" in rho
    assert "--ipopt-use-sx" in rho
    assert "--ipopt-enable-periodic-fes-warmup-projection" in rho
    assert rho[rho.index("--periodic-fes-warmup-projection-strategy") + 1] == "rollout"
    assert "--common-initial-solution-recenter-first-node-bounds" in rho
    assert "--ipopt-no-use-sx" not in rho
    assert rho[rho.index("--ipopt-max-iter") + 1] == "2000"
    assert "--single-shot" in full
    assert full[full.index("--mechanical-formulation") + 1] == "reduced"
    assert full[full.index("--full-horizon-prefix-solution") + 1] == str(
        tmp_path / "previous-full.npz"
    )
    assert (
        paired_reduced[paired_reduced.index("--mechanical-formulation") + 1]
        == "reduced"
    )
    assert "--ipopt-no-use-sx" in paired_reduced
    assert full[full.index("--madnlp-linear-solver") + 1] == "mumps"
    assert "--ipopt-no-use-sx" in full
    assert "--ipopt-disable-standard-warmup" in full
    assert "--adopt-common-initial-solution-warmup-cycles" in full
    assert "--ipopt-disable-standard-warmup" not in one_cycle_full
    assert "--adopt-common-initial-solution-warmup-cycles" not in one_cycle_full
    assert "--optional-nlp-periodic-ipopt-hot-start" in full
    assert "--initial-guess-diagnostics" in full
    assert "--exact-initial-nlp-audit" in full
    assert "--acados-diagnostics" not in full
    assert "--periodic-ipopt-refinement-use-sx" in full
    assert full[full.index("--periodic-ipopt-refinement-iterations") + 1] == "2000"


def test_full_horizon_can_use_ipopt_locally_with_mx(tmp_path):
    args = SimpleNamespace(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=3,
        full_horizon_solver="ipopt",
    )

    command = full_horizon._full_horizon_command(
        args,
        3,
        tmp_path / "prefix.npz",
        tmp_path / "full.json",
        tmp_path / "full.npz",
    )

    assert command[command.index("--solvers") + 1] == "ipopt"
    assert "--single-shot" in command
    assert "--ipopt-no-use-sx" in command
    assert command[command.index("--ipopt-linear-solver") + 1] == "mumps"
    assert "--exact-initial-nlp-audit" not in command
