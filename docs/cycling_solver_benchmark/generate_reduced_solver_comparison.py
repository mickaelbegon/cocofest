#!/usr/bin/env python3
"""Compare executed reduced-RHO trajectories exported by the solver benchmark.

The script intentionally consumes only the benchmark JSON/NPZ artifacts.  It
does not import cocofest or Bioptim, which makes the post-processing independent
of the solver environments used to produce each trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MUSCLES = ("Biceps", "Triceps", "Delt_ant", "Delt_post")
COLORS = {"IPOPT R5": "#3366cc", "MadNLP R5": "#dc3912", "ACADOS + IPOPT": "#109618"}
DEFAULT_SOURCE_RUNS = {
    "IPOPT R5": 31380186719,
    "MadNLP R5": 31380186719,
    "ACADOS + IPOPT": 31428024125,
}
PW_MIN_US = 131.405
PW_MAX_US = 600.0


def _load_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload["results"][0] if "results" in payload else payload


def _load_trajectory(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key], dtype=float).reshape(-1)
            for key in archive.files
            if key != "metadata__json"
        }
        metadata = json.loads(str(archive["metadata__json"]))
    return arrays, metadata


def _controls(arrays: dict[str, np.ndarray], cycles: int) -> dict[str, np.ndarray]:
    output = {}
    for muscle in MUSCLES:
        values = arrays[f"controls__last_pulse_width_{muscle}"] * 1e6
        if values.size < cycles * 30:
            raise ValueError(f"Only {values.size // 30} control cycles available for {muscle}")
        output[muscle] = values[: cycles * 30].reshape(cycles, 30)
    return output


def _capacity_traces(
    arrays: dict[str, np.ndarray], result: dict, cycles: int
) -> dict[str, np.ndarray]:
    output = {}
    for muscle in MUSCLES:
        values = arrays[f"states__A_{muscle}"]
        exported_cycles = int(
            result.get("exported_cycles") or result.get("validated_cycles") or cycles
        )
        if (values.size - 1) % exported_cycles:
            raise ValueError(
                f"State trace for {muscle} cannot be split into {exported_cycles} cycles"
            )
        points_per_cycle = (values.size - 1) // exported_cycles
        boundaries = values[: cycles * points_per_cycle + 1 : points_per_cycle]
        scale = float(result["fatigue_capacity_scales"][f"A_{muscle}"])
        output[muscle] = boundaries / scale
    return output


def _dense_capacity_traces(
    arrays: dict[str, np.ndarray], result: dict, cycles: int
) -> dict[str, np.ndarray]:
    """Return every exported integration/collocation sample up to ``cycles``."""

    output = {}
    for muscle in MUSCLES:
        values = arrays[f"states__A_{muscle}"]
        exported_cycles = int(
            result.get("exported_cycles") or result.get("validated_cycles") or cycles
        )
        if (values.size - 1) % exported_cycles:
            raise ValueError(
                f"State trace for {muscle} cannot be split into {exported_cycles} cycles"
            )
        points_per_cycle = (values.size - 1) // exported_cycles
        scale = float(result["fatigue_capacity_scales"][f"A_{muscle}"])
        output[muscle] = values[: cycles * points_per_cycle + 1] / scale
    return output


def _trapz_cycle(values: np.ndarray, cycles: int) -> float:
    if values.size < 2:
        return 0.0
    dx = float(cycles) / (values.size - 1)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * dx))


def _fatigue_metrics(capacities: dict[str, np.ndarray], cycles: int) -> dict:
    per_muscle = {}
    for muscle, capacity in capacities.items():
        fatigue = 1.0 - capacity
        per_muscle[muscle] = {
            "initial_capacity_ratio": float(capacity[0]),
            "final_capacity_ratio": float(capacity[-1]),
            "fatigue_auc_cycles": _trapz_cycle(fatigue, cycles),
            "executed_fatigue_objective": 10_000.0 * _trapz_cycle(fatigue**2, cycles),
        }
    return {
        "per_muscle": per_muscle,
        "fatigue_auc_cycles": float(sum(row["fatigue_auc_cycles"] for row in per_muscle.values())),
        "executed_fatigue_objective": float(
            sum(row["executed_fatigue_objective"] for row in per_muscle.values())
        ),
        "min_final_capacity_ratio": float(
            min(row["final_capacity_ratio"] for row in per_muscle.values())
        ),
    }


def _timing_metrics(result: dict, cycles: int) -> dict:
    windows = result["windows"][:cycles]
    solver = np.asarray([float(row["solver_time_s"]) for row in windows])
    wall = np.asarray([float(row["wall_time_s"]) for row in windows])
    hot_solver = solver[1:] if solver.size > 1 else solver
    hot_wall = wall[1:] if wall.size > 1 else wall
    return {
        "online_solver_total_s": float(np.sum(solver)),
        "online_wall_total_s": float(np.sum(wall)),
        "online_solver_mean_s_per_rho": float(np.mean(solver)),
        "online_wall_mean_s_per_rho": float(np.mean(wall)),
        "hot_solver_median_s": float(np.median(hot_solver)),
        "hot_solver_p90_s": float(np.percentile(hot_solver, 90)),
        "hot_wall_median_s": float(np.median(hot_wall)),
        "hot_wall_p90_s": float(np.percentile(hot_wall, 90)),
        "reported_full_run_end_to_end_s": float(result["end_to_end_wall_time_s"]),
        "reported_full_run_cycles": int(result.get("validated_cycles") or len(result["windows"])),
    }


def _hybrid_fallback_metrics(result: dict, cycles: int) -> dict:
    """Account for IPOPT work attempted by the ACADOS hybrid chain."""

    summaries = [
        row
        for row in result.get("acados_ipopt_recovery_summaries", [])
        if int(row.get("target_rho") or row.get("attempt_window") or 0) <= cycles
    ]
    return {
        "recovery_attempt_count": len(summaries),
        "recovery_accepted_count": sum(bool(row.get("accepted")) for row in summaries),
        "fallback_advanced_count": sum(
            bool(row.get("fallback_advanced")) for row in summaries
        ),
        "recovery_solver_total_s": float(
            sum(float(row.get("solver_time_s") or 0.0) for row in summaries)
        ),
        "recovery_wall_total_s": float(
            sum(float(row.get("wall_time_s") or 0.0) for row in summaries)
        ),
    }


def _control_metrics(controls: dict[str, np.ndarray]) -> dict:
    rows = {}
    for muscle, pw in controls.items():
        rows[muscle] = {
            "mean_us": float(np.mean(pw)),
            "standard_deviation_us": float(np.std(pw)),
            "minimum_us": float(np.min(pw)),
            "maximum_us": float(np.max(pw)),
            "fraction_at_lower_bound": float(np.mean(pw <= PW_MIN_US + 0.01)),
            "fraction_at_upper_bound": float(np.mean(pw >= PW_MAX_US - 0.01)),
        }
    return rows


def _pairwise_control_metrics(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict:
    rows = {}
    for muscle in MUSCLES:
        a, b = left[muscle].reshape(-1), right[muscle].reshape(-1)
        rows[muscle] = {
            "mae_us": float(np.mean(np.abs(a - b))),
            "rmse_us": float(np.sqrt(np.mean((a - b) ** 2))),
            "maximum_absolute_error_us": float(np.max(np.abs(a - b))),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
        }
    return rows


def _plot_pw_profiles(data: dict, cycles: int, output: Path) -> None:
    selected = [1, min(30, cycles), min(100, cycles), cycles]
    selected = list(dict.fromkeys(selected))
    phase = np.arange(30) * 360.0 / 30.0
    fig, axes = plt.subplots(
        len(MUSCLES), len(selected), figsize=(14, 10), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for row, muscle in enumerate(MUSCLES):
        for col, cycle in enumerate(selected):
            ax = axes[row, col]
            for solver, item in data.items():
                ax.step(
                    phase,
                    item["controls"][muscle][cycle - 1],
                    where="post",
                    color=COLORS[solver],
                    linewidth=1.35,
                    label=solver,
                )
            ax.axhline(PW_MIN_US, color="0.65", linestyle=":", linewidth=0.8)
            ax.axhline(PW_MAX_US, color="0.65", linestyle=":", linewidth=0.8)
            if row == 0:
                ax.set_title(f"RHO {cycle}")
            if col == 0:
                ax.set_ylabel(f"{muscle}\nPW (µs)")
            if row == len(MUSCLES) - 1:
                ax.set_xlabel("Phase du pédalier (°)")
            ax.set_xlim(0, 348)
            ax.set_ylim(115, 615)
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Contrôles exécutés — formulation reduced, 30 stimulations/cycle", y=1.025
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_control_differences(data: dict, output: Path) -> None:
    reference = data["IPOPT R5"]["controls"]
    comparators = ("MadNLP R5", "ACADOS + IPOPT")
    fig, axes = plt.subplots(len(MUSCLES), 2, figsize=(13, 9), sharex=True, sharey=True)
    image = None
    for row, muscle in enumerate(MUSCLES):
        max_abs = max(
            np.max(np.abs(data[name]["controls"][muscle] - reference[muscle]))
            for name in comparators
        )
        max_abs = max(max_abs, 1.0)
        for col, name in enumerate(comparators):
            delta = data[name]["controls"][muscle] - reference[muscle]
            image = axes[row, col].imshow(
                delta,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-max_abs,
                vmax=max_abs,
                extent=(0, 360, 1, delta.shape[0]),
            )
            if row == 0:
                axes[row, col].set_title(f"{name} − IPOPT R5")
            if col == 0:
                axes[row, col].set_ylabel(f"{muscle}\nRHO")
            if row == len(MUSCLES) - 1:
                axes[row, col].set_xlabel("Phase du pédalier (°)")
            fig.colorbar(image, ax=axes[row, col], label="ΔPW (µs)", pad=0.01)
    fig.suptitle("Écarts des contrôles exécutés par rapport à IPOPT", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fatigue(data: dict, cycles: int, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(cycles + 1)
    for muscle in MUSCLES:
        for solver, item in data.items():
            axes[0].plot(
                x,
                item["capacities"][muscle],
                color=COLORS[solver],
                linestyle={
                    "Biceps": "-",
                    "Triceps": "--",
                    "Delt_ant": "-.",
                    "Delt_post": ":",
                }[muscle],
                linewidth=1.25,
            )
    axes[0].set_xlabel("Cycles exécutés")
    axes[0].set_ylabel("Capacité $A/A_{scale}$")
    axes[0].set_title("Évolution des capacités")
    axes[0].grid(alpha=0.2)

    width = 0.25
    positions = np.arange(len(MUSCLES))
    for index, (solver, item) in enumerate(data.items()):
        finals = [
            item["fatigue"]["per_muscle"][muscle]["final_capacity_ratio"]
            for muscle in MUSCLES
        ]
        axes[1].bar(
            positions + (index - 1) * width,
            finals,
            width,
            color=COLORS[solver],
            label=solver,
        )
    axes[1].set_xticks(positions, MUSCLES, rotation=20)
    axes[1].set_ylim(0.84, 1.005)
    axes[1].set_ylabel("Capacité finale $A/A_{scale}$")
    axes[1].set_title(f"Capacité après {cycles} RHO")
    axes[1].grid(axis="y", alpha=0.2)

    solver_handles = [plt.Line2D([0], [0], color=COLORS[name], lw=3, label=name) for name in data]
    muscle_handles = [
        plt.Line2D([0], [0], color="0.25", lw=1.5, linestyle=style, label=muscle)
        for muscle, style in zip(MUSCLES, ("-", "--", "-.", ":"))
    ]
    all_handles = solver_handles + muscle_handles
    fig.legend(
        all_handles,
        [handle.get_label() for handle in all_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timing(data: dict, cycles: int, output: Path) -> None:
    names = list(data)
    medians = [data[name]["timing"]["hot_solver_median_s"] for name in names]
    p90s = [data[name]["timing"]["hot_solver_p90_s"] for name in names]
    totals = [data[name]["timing"]["online_wall_total_s"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(names))
    axes[0].bar(
        x - 0.18,
        medians,
        0.36,
        label="médiane",
        color=[COLORS[name] for name in names],
        alpha=0.8,
    )
    axes[0].bar(
        x + 0.18,
        p90s,
        0.36,
        label="P90",
        color=[COLORS[name] for name in names],
        hatch="//",
        alpha=0.45,
    )
    axes[0].set_xticks(x, names, rotation=15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Temps solveur chaud / RHO (s, log)")
    axes[0].set_title("Latence en ligne")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    bars = axes[1].bar(x, totals, color=[COLORS[n] for n in names], alpha=0.85)
    axes[1].set_xticks(x, names, rotation=15)
    axes[1].set_ylabel("Temps mur-à-mur en ligne cumulé (s)")
    axes[1].set_title(f"Somme des {cycles} fenêtres")
    axes[1].bar_label(bars, fmt="%.1f s", padding=3)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=145)
    parser.add_argument("--ipopt-result", type=Path, required=True)
    parser.add_argument("--ipopt-trajectory", type=Path, required=True)
    parser.add_argument("--madnlp-result", type=Path, required=True)
    parser.add_argument("--madnlp-trajectory", type=Path, required=True)
    parser.add_argument("--acados-result", type=Path, required=True)
    parser.add_argument("--acados-trajectory", type=Path, required=True)
    parser.add_argument(
        "--ipopt-run-id", type=int, default=DEFAULT_SOURCE_RUNS["IPOPT R5"]
    )
    parser.add_argument(
        "--madnlp-run-id", type=int, default=DEFAULT_SOURCE_RUNS["MadNLP R5"]
    )
    parser.add_argument(
        "--acados-run-id", type=int, default=DEFAULT_SOURCE_RUNS["ACADOS + IPOPT"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications = {
        "IPOPT R5": (args.ipopt_result, args.ipopt_trajectory),
        "MadNLP R5": (args.madnlp_result, args.madnlp_trajectory),
        "ACADOS + IPOPT": (args.acados_result, args.acados_trajectory),
    }
    source_runs = {
        "IPOPT R5": args.ipopt_run_id,
        "MadNLP R5": args.madnlp_run_id,
        "ACADOS + IPOPT": args.acados_run_id,
    }
    data = {}
    for name, (result_path, trajectory_path) in specifications.items():
        result = _load_result(result_path)
        arrays, metadata = _load_trajectory(trajectory_path)
        controls = _controls(arrays, args.cycles)
        capacities = _capacity_traces(arrays, result, args.cycles)
        dense_capacities = _dense_capacity_traces(arrays, result, args.cycles)
        data[name] = {
            "controls": controls,
            "capacities": capacities,
            # Fatigue quadrature uses every exported collocation/integration
            # sample.  Boundary-only traces remain preferable for figures so
            # Radau-5 and IRK are plotted on the same one-point-per-RHO grid.
            "fatigue": _fatigue_metrics(dense_capacities, args.cycles),
            "timing": _timing_metrics(result, args.cycles),
            "hybrid_fallback": _hybrid_fallback_metrics(result, args.cycles),
            "metadata": metadata,
            "control_summary": _control_metrics(controls),
            "source_github_actions_run": source_runs[name],
            "source_result": result_path.name,
            "source_trajectory": trajectory_path.name,
        }

    summary = {
        "cycles_compared": args.cycles,
        "stimulations_per_cycle": 30,
        "pulse_width_bounds_us": [PW_MIN_US, PW_MAX_US],
        "solvers": {
            name: {
                key: value
                for key, value in item.items()
                if key not in {"controls", "capacities"}
            }
            for name, item in data.items()
        },
        "pairwise_control_differences": {
            "MadNLP R5 vs IPOPT R5": _pairwise_control_metrics(
                data["IPOPT R5"]["controls"], data["MadNLP R5"]["controls"]
            ),
            "ACADOS + IPOPT vs IPOPT R5": _pairwise_control_metrics(
                data["IPOPT R5"]["controls"], data["ACADOS + IPOPT"]["controls"]
            ),
            "ACADOS + IPOPT vs MadNLP R5": _pairwise_control_metrics(
                data["MadNLP R5"]["controls"], data["ACADOS + IPOPT"]["controls"]
            ),
        },
    }
    (args.output_dir / "reduced_solver_comparison_145.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot_pw_profiles(data, args.cycles, args.output_dir / "reduced_solver_pw_profiles_145.png")
    _plot_control_differences(data, args.output_dir / "reduced_solver_pw_differences_145.png")
    _plot_fatigue(data, args.cycles, args.output_dir / "reduced_solver_fatigue_145.png")
    _plot_timing(data, args.cycles, args.output_dir / "reduced_solver_timing_145.png")


if __name__ == "__main__":
    main()
