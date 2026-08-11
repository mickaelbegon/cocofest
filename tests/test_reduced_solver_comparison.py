from docs.cycling_solver_benchmark.generate_reduced_solver_comparison import (
    _timing_metrics,
)


def test_hybrid_timing_includes_failed_target_attempts_and_ipopt_recovery():
    result = {
        "validated_cycles": 3,
        "end_to_end_wall_time_s": 180.0,
        "windows": [],
        "solver_attempt_accounting": {
            "attempts": [
                {"target_rho": 1, "solver_time_s": 1.0, "wall_time_s": 10.0},
                {"target_rho": 2, "solver_time_s": 2.0, "wall_time_s": 20.0},
                {"target_rho": 2, "solver_time_s": 3.0, "wall_time_s": 30.0},
                {"target_rho": 3, "solver_time_s": 4.0, "wall_time_s": 40.0},
            ]
        },
        "acados_ipopt_recovery_summaries": [
            {"target_rho": 2, "solver_time_s": 5.0, "wall_time_s": 50.0}
        ],
        "execution_timing": {"rho_solve_loop_wall_time_s": 175.0},
    }

    timing = _timing_metrics(result, cycles=3)

    assert timing["online_solver_total_s"] == 15.0
    assert timing["online_wall_total_s"] == 150.0
    assert timing["hot_solver_median_s"] == 7.0
    assert timing["hot_wall_median_s"] == 70.0
    assert timing["target_solver_attempt_count"] == 4
    assert timing["recovery_attempt_count"] == 1
    assert timing["rho_pipeline_wall_total_s"] == 175.0


def test_legacy_window_timing_remains_supported_for_a_prefix():
    result = {
        "validated_cycles": 3,
        "end_to_end_wall_time_s": 12.0,
        "windows": [
            {"solver_time_s": 1.0, "wall_time_s": 2.0},
            {"solver_time_s": 2.0, "wall_time_s": 3.0},
            {"solver_time_s": 9.0, "wall_time_s": 10.0},
        ],
        "execution_timing": {"rho_solve_loop_wall_time_s": 11.0},
    }

    timing = _timing_metrics(result, cycles=2)

    assert timing["online_solver_total_s"] == 3.0
    assert timing["online_wall_total_s"] == 5.0
    assert timing["hot_solver_median_s"] == 2.0
    assert timing["rho_pipeline_wall_total_s"] is None
