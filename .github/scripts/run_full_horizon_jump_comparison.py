#!/usr/bin/env python3
"""Compare three sequential FHO increments with one multi-cycle FHO jump."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


RUNNER_PATH = Path(__file__).with_name("run_full_horizon_benchmark.py")
SPEC = importlib.util.spec_from_file_location("full_horizon_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cycles", type=int, required=True)
    parser.add_argument("--jump-cycles", type=int, default=3)
    parser.add_argument("--memory-limit-gib", default="auto")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--full-horizon-solver", choices=("ipopt", "madnlp"), default="ipopt")
    parser.add_argument("--crank-assistance", type=float, default=0.0)
    parser.add_argument("--terminal-wheel-q-slack", type=float, default=0.002)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--attempt-timeout-s", type=float, default=21600.0)
    return parser


def _successful_by_cycle(records: list[dict], key: str) -> dict[int, dict]:
    return {
        int(record[key]): record
        for record in records
        if record.get("success") is True
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.workspace = args.workspace.resolve()
    args.seed_dir = args.seed_dir.resolve()
    args.source_output_dir = args.source_output_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.jump_cycles < 2:
        raise ValueError("--jump-cycles must be at least two.")
    target_cycles = args.baseline_cycles + args.jump_cycles
    args.max_cycles = target_cycles
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_report = json.loads(
        (args.source_output_dir / "full-horizon-report.json").read_text(encoding="utf-8")
    )
    rho_seed = Path(source_report["rho"]["seed_path"])
    fho_by_cycle = _successful_by_cycle(
        source_report.get("full_horizon_attempts", []), "cycles"
    )
    extension_by_cycle = _successful_by_cycle(
        source_report.get("extension_rho_attempts", []), "target_cycle"
    )
    if args.baseline_cycles not in fho_by_cycle:
        raise ValueError("The requested baseline FHO is not certified.")
    baseline_solution = Path(fho_by_cycle[args.baseline_cycles]["solution_path"])
    if not baseline_solution.is_file() or not rho_seed.is_file():
        raise FileNotFoundError("The baseline FHO or concatenated RHO seed is missing.")

    total_memory = runner.available_memory_bytes()
    rss_limit_gib = runner._parse_memory_limit(args.memory_limit_gib, total_memory)
    rss_limit_bytes = int(rss_limit_gib * runner.GIB)
    concatenated_seed = baseline_solution
    continuation_terminal = baseline_solution
    jump_extensions = []
    for target in range(args.baseline_cycles + 1, target_cycles + 1):
        extension = runner._run_extension_rho(
            args,
            source_full_solution=continuation_terminal,
            reference_rho_seed=rho_seed,
            after_cycles=target - 1,
            rss_limit_bytes=rss_limit_bytes,
        )
        jump_extensions.append(extension)
        if not extension["success"]:
            break
        next_seed = args.output_dir / "jump-seeds" / f"fho-{args.baseline_cycles:04d}-plus-rho-{target:04d}.npz"
        runner.append_rho_extension_cycle(
            concatenated_seed,
            Path(extension["solution_path"]),
            next_seed,
        )
        concatenated_seed = next_seed
        continuation_terminal = Path(extension["solution_path"])

    jump_attempt = None
    if len(jump_extensions) == args.jump_cycles and all(
        extension["success"] for extension in jump_extensions
    ):
        jump_attempt = runner._run_horizon_attempt(
            args,
            rho_seed=concatenated_seed,
            cycles=target_cycles,
            phase=f"jump_{args.jump_cycles}",
            chance=1,
            rss_limit_bytes=rss_limit_bytes,
            prefix_solution_path=baseline_solution,
            heartbeat_seed_label=(
                f"FHO_{args.baseline_cycles}+"
                + "+".join(
                    f"RHO_{cycle}"
                    for cycle in range(args.baseline_cycles + 1, target_cycles + 1)
                )
            ),
        )

    sequential_records = []
    for target in range(args.baseline_cycles + 1, target_cycles + 1):
        if target not in extension_by_cycle or target not in fho_by_cycle:
            sequential_records = []
            break
        sequential_records.append(
            {
                "target_cycle": target,
                "extension_elapsed_s": extension_by_cycle[target]["elapsed_s"],
                "fho_elapsed_s": fho_by_cycle[target]["elapsed_s"],
            }
        )
    sequential_elapsed_s = (
        sum(row["extension_elapsed_s"] + row["fho_elapsed_s"] for row in sequential_records)
        if sequential_records
        else None
    )
    jump_elapsed_s = sum(row["elapsed_s"] for row in jump_extensions) + (
        0.0 if jump_attempt is None else jump_attempt["elapsed_s"]
    )
    comparison = {
        "baseline_cycles": args.baseline_cycles,
        "target_cycles": target_cycles,
        "jump_cycles": args.jump_cycles,
        "rss_limit_gib": rss_limit_gib,
        "sequential_records": sequential_records,
        "sequential_elapsed_s": sequential_elapsed_s,
        "jump_extensions": jump_extensions,
        "jump_attempt": jump_attempt,
        "jump_elapsed_s": jump_elapsed_s,
        "speedup_vs_sequential": (
            None
            if sequential_elapsed_s is None or jump_elapsed_s <= 0.0
            else sequential_elapsed_s / jump_elapsed_s
        ),
    }
    output = args.output_dir / "jump-comparison.json"
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    print(f"jump comparison: {output}")
    return 0 if jump_attempt and jump_attempt.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
