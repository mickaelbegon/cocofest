#!/usr/bin/env python3
"""Run an RSS-bounded RHO-to-full-horizon continuation with MadNLP.

Two consecutive reduced RHO cycles first initialize FHO_2.  Every subsequent
problem is then built from the last certified full-horizon solution: its
terminal state initializes one new reduced RHO cycle, and the concatenation
``FHO_N + RHO_(N+1)`` initializes FHO_(N+1).  The horizon therefore grows by
exactly one cycle and never reuses an unrelated tail from the original RHO.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

GIB = 1024**3
SMALL_RUNNER_RSS_LIMIT_GIB = 12.5
LARGE_RUNNER_RSS_LIMIT_GIB = 97.5
RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP = 1.0 / 256.0


def horizon_sweep_targets(max_cycles: int) -> list[int]:
    """Return the strict one-cycle-at-a-time FHO continuation ladder."""

    if max_cycles < 2:
        raise ValueError("max_cycles must be at least two.")
    return list(range(2, max_cycles + 1))


def refinement_targets(last_success: int, first_failure: int) -> list[int]:
    """Fill the final coarse interval without retrying either endpoint."""

    if last_success < 0 or first_failure <= last_success:
        raise ValueError("The refinement interval must be ordered.")
    return list(range(last_success + 1, first_failure))


def automatic_rss_limit_gib(total_memory_bytes: int) -> float:
    """Choose a conservative RSS cap for 16 GiB and 128 GiB runners."""

    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be strictly positive.")
    total_gib = total_memory_bytes / GIB
    if total_gib <= 32.0:
        return min(SMALL_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    if total_gib >= 96.0:
        return min(LARGE_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    # Intermediate machines are not benchmark targets, but retaining roughly
    # 22 % headroom is safer than extrapolating the 128 GiB absolute cap.
    return 0.78 * total_gib


def _read_positive_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def available_memory_bytes() -> int:
    """Return the tighter physical/cgroup allocation, with a macOS fallback."""

    physical = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                physical = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_limits = [
        _read_positive_integer(Path("/sys/fs/cgroup/memory.max")),
        _read_positive_integer(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    candidates = [value for value in (physical, *cgroup_limits) if value]
    if not candidates:
        try:
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if page_count > 0 and page_size > 0:
                candidates.append(page_count * page_size)
        except (OSError, ValueError, TypeError):
            pass
    if not candidates:
        raise RuntimeError("Cannot determine the runner memory allocation.")
    # Some cgroup v1 hosts expose an effectively infinite sentinel.
    finite = [value for value in candidates if value < (1 << 60)]
    return min(finite or candidates)


def _child_pids(pid: int) -> tuple[int, ...]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return tuple(int(value) for value in children_path.read_text().split())
    except (OSError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["pgrep", "-P", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        return tuple(int(value) for value in completed.stdout.split())
    except (OSError, ValueError):
        return ()


def process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        pending.extend(_child_pids(pid))
    return observed


def _process_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return 0
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        rss_kib = int(completed.stdout.strip())
        return rss_kib * 1024 if rss_kib > 0 else 0
    except (OSError, ValueError):
        pass
    return 0


def process_tree_rss_bytes(root_pid: int) -> int:
    return sum(_process_rss_bytes(pid) for pid in process_tree_pids(root_pid))


@dataclass
class MonitoredRun:
    command: list[str]
    return_code: int
    peak_rss_bytes: int
    elapsed_s: float
    memory_limit_exceeded: bool
    timed_out: bool
    log_path: str


def _terminate_process_group(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_monitored(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    rss_limit_bytes: int,
    poll_interval_s: float = 0.5,
    timeout_s: float | None = None,
) -> MonitoredRun:
    """Run one solver process and stop its whole process group at the RSS cap."""

    if rss_limit_bytes <= 0:
        raise ValueError("rss_limit_bytes must be strictly positive.")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be strictly positive when provided.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak_rss = 0
    memory_limit_exceeded = False
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_heartbeat = start + 30.0
        while process.poll() is None:
            rss = process_tree_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if rss >= rss_limit_bytes:
                memory_limit_exceeded = True
                message = (
                    f"RSS limit reached: {rss / GIB:.3f} GiB >= "
                    f"{rss_limit_bytes / GIB:.3f} GiB"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            now = time.monotonic()
            if timeout_s is not None and now - start >= timeout_s:
                timed_out = True
                message = f"Attempt timeout reached after {now - start:.1f} s"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            if now >= next_heartbeat:
                print(
                    f"full-horizon heartbeat: pid={process.pid} "
                    f"rss={rss / GIB:.3f} GiB peak={peak_rss / GIB:.3f} GiB",
                    flush=True,
                )
                next_heartbeat = now + 30.0
            time.sleep(poll_interval_s)
        return_code = process.wait()
        peak_rss = max(peak_rss, process_tree_rss_bytes(process.pid))

    return MonitoredRun(
        command=command,
        return_code=return_code,
        peak_rss_bytes=peak_rss,
        elapsed_s=time.monotonic() - start,
        memory_limit_exceeded=memory_limit_exceeded,
        timed_out=timed_out,
        log_path=str(log_path),
    )


def _load_metadata(data) -> dict:
    if "metadata__json" not in data.files:
        raise ValueError("The RHO seed has no metadata__json entry.")
    return json.loads(str(data["metadata__json"].item()))


def write_rho_seed_prefix(
    source_path: Path, output_path: Path, target_cycles: int
) -> dict:
    """Slice a concatenated RHO seed to an exact multi-cycle prefix."""

    if target_cycles < 1:
        raise ValueError("target_cycles must be strictly positive.")
    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if target_cycles > source_cycles:
            raise ValueError(
                f"Cannot extract {target_cycles} cycles from {source_cycles}."
            )
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder:
                    raise ValueError(
                        f"State seed '{key}' cannot be divided into "
                        f"{source_cycles} cycles."
                    )
                payload[key] = values[..., : target_cycles * intervals + 1]
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder:
                    raise ValueError(
                        f"Control seed '{key}' cannot be divided into "
                        f"{source_cycles} cycles."
                    )
                payload[key] = values[..., : target_cycles * nodes]
            else:
                payload[key] = values
    metadata.update(
        {
            "cycles_per_window": target_cycles,
            "producer_mode": "receding_horizon_prefix",
            "producer_source_cycles": source_cycles,
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_rho_seed_cycle(
    source_path: Path, output_path: Path, cycle_number: int
) -> dict:
    """Extract one one-based cycle from a concatenated reduced RHO trace."""

    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if cycle_number < 1 or cycle_number > source_cycles:
            raise ValueError(
                f"cycle_number must be in [1, {source_cycles}], got {cycle_number}."
            )
        cycle_index = cycle_number - 1
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder:
                    raise ValueError(f"State seed '{key}' has an invalid cycle layout.")
                start = cycle_index * intervals
                payload[key] = values[..., start : start + intervals + 1].copy()
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder:
                    raise ValueError(
                        f"Control seed '{key}' has an invalid cycle layout."
                    )
                start = cycle_index * nodes
                payload[key] = values[..., start : start + nodes].copy()
            else:
                payload[key] = values
    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "receding_horizon_cycle_extraction",
            "producer_source_cycles": source_cycles,
            "producer_cycle_number": cycle_number,
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_rho_initial_state_homotopy_seed(
    source_solution_path: Path,
    reference_cycle_path: Path,
    target_fho_path: Path,
    output_path: Path,
    fraction: float,
) -> dict:
    """Move a one-cycle RHO warm start toward the terminal state of FHO_N."""

    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and lie in [0, 1].")
    with np.load(source_solution_path, allow_pickle=False) as source, np.load(
        reference_cycle_path, allow_pickle=False
    ) as reference, np.load(target_fho_path, allow_pickle=False) as target:
        source_metadata = _load_metadata(source)
        reference_metadata = _load_metadata(reference)
        target_metadata = _load_metadata(target)
        if any(
            metadata.get("mechanical_formulation") != "reduced"
            for metadata in (source_metadata, reference_metadata, target_metadata)
        ):
            raise ValueError(
                "The RHO initial-state homotopy requires reduced mechanics."
            )
        if (
            int(source_metadata["cycles_per_window"]) != 1
            or int(reference_metadata["cycles_per_window"]) != 1
        ):
            raise ValueError(
                "The homotopy source and reference must contain one cycle."
            )

        source_keys = set(source.files) - {"metadata__json"}
        reference_keys = set(reference.files) - {"metadata__json"}
        if source_keys != reference_keys or not source_keys <= set(target.files):
            raise ValueError(
                "The homotopy checkpoints do not expose matching variables."
            )

        payload: dict[str, np.ndarray] = {}
        maximum_initial_state_change = 0.0
        theta_winding_shift = 0.0
        for key in sorted(source_keys):
            source_values = np.asarray(source[key])
            reference_values = np.asarray(reference[key])
            if source_values.shape != reference_values.shape:
                raise ValueError(f"Homotopy variable '{key}' has incompatible shapes.")
            if key.startswith("states__"):
                target_values = np.asarray(target[key])
                source_for_homotopy = source_values
                reference_for_homotopy = reference_values
                if key == "states__theta":
                    target_initial = target_values[..., -1:]
                    winding_turns = np.rint(
                        (target_initial - reference_values[..., :1]) / (2.0 * np.pi)
                    )
                    candidate_shift = winding_turns * (2.0 * np.pi)
                    residual = (
                        target_initial
                        - reference_values[..., :1]
                        - candidate_shift
                    )
                    if float(np.max(np.abs(residual))) <= 0.05:
                        reference_for_homotopy = reference_values + candidate_shift
                        source_shift = np.rint(
                            (
                                reference_for_homotopy[..., :1]
                                - source_values[..., :1]
                            )
                            / (2.0 * np.pi)
                        ) * (2.0 * np.pi)
                        source_for_homotopy = source_values + source_shift
                        theta_winding_shift = float(candidate_shift.reshape(-1)[0])
                desired_initial = reference_for_homotopy[..., :1] + fraction * (
                    target_values[..., -1:] - reference_for_homotopy[..., :1]
                )
                change = desired_initial - source_for_homotopy[..., :1]
                maximum_initial_state_change = max(
                    maximum_initial_state_change,
                    float(np.max(np.abs(change))),
                )
                if key == "states__theta":
                    shifted = source_for_homotopy + change
                else:
                    shifted = source_for_homotopy + change * np.linspace(
                        1.0, 0.0, source_values.shape[-1]
                    )
                shifted[..., 0] = desired_initial[..., 0]
                payload[key] = shifted
            else:
                payload[key] = source_values.copy()

    metadata = dict(source_metadata)
    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "rho_initial_state_homotopy",
            "homotopy_fraction": float(fraction),
            "homotopy_reference_cycle_number": reference_metadata.get(
                "producer_cycle_number"
            ),
            "homotopy_target_fho_cycles": target_metadata.get("cycles_per_window"),
            "homotopy_maximum_initial_state_change": maximum_initial_state_change,
            "homotopy_theta_winding_shift": theta_winding_shift,
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def write_fho_terminal_continuation_seed(source_path: Path, output_path: Path) -> dict:
    """Extrapolate the last FHO cycle into a one-cycle seed starting at its end.

    The state construction matches the established one-cycle tiling policy in
    the cycling example.  In particular, the first node is copied exactly from
    the FHO terminal state; the actual reduced RHO solve restores dynamics and
    physical feasibility before this cycle may seed a longer FHO.
    """

    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if source_cycles < 1:
            raise ValueError("The full-horizon source must contain a cycle.")
        if metadata.get("mechanical_formulation") != "reduced":
            raise ValueError("The continuation source must use reduced mechanics.")

        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                state_key = key.split("__", 1)[1]
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder or intervals < 1:
                    raise ValueError(
                        f"State solution '{key}' has an invalid cycle layout."
                    )
                last_cycle = np.asarray(values[..., -intervals - 1 :], dtype=float)
                drift = last_cycle[:, -1:] - last_cycle[:, :1]
                continuation = np.empty_like(last_cycle)
                continuation[:, :1] = last_cycle[:, -1:]
                continuation[:, 1:] = last_cycle[:, 1:]
                if state_key == "theta":
                    continuation[:, 1:] += drift
                elif state_key == "q":
                    continuation[-1:, 1:] += drift[-1:, :]
                    if continuation.shape[0] > 1:
                        continuation[:-1, 1:] += drift[:-1, :] * np.linspace(
                            1.0, 0.0, intervals
                        )
                elif state_key.startswith(("F_", "A_", "Tau1_", "Km_")):
                    continuation[:, 1:] += drift
                else:
                    continuation[:, 1:] += drift * np.linspace(1.0, 0.0, intervals)
                payload[key] = continuation
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder or nodes < 1:
                    raise ValueError(
                        f"Control solution '{key}' has an invalid cycle layout."
                    )
                payload[key] = np.asarray(values[..., -nodes:]).copy()
            else:
                payload[key] = values

    metadata.update(
        {
            "cycles_per_window": 1,
            "producer_mode": "full_horizon_terminal_continuation",
            "producer_source_cycles": source_cycles,
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def append_rho_extension_cycle(
    prefix_path: Path,
    extension_path: Path,
    output_path: Path,
) -> dict:
    """Append one terminal-state RHO to the reduced seed behind FHO_N.

    The old reduced prefix is only a shape carrier: FHO_N will overwrite it.
    Its last state node is replaced with the extension's initial state so the
    boundary inspected before that overlay already represents the true
    FHO_N-to-RHO_(N+1) seam.
    """

    with np.load(prefix_path, allow_pickle=False) as prefix_data, np.load(
        extension_path, allow_pickle=False
    ) as extension_data:
        prefix_metadata = _load_metadata(prefix_data)
        extension_metadata = _load_metadata(extension_data)
        prefix_cycles = int(prefix_metadata["cycles_per_window"])
        if int(extension_metadata["cycles_per_window"]) != 1:
            raise ValueError("The RHO extension must contain exactly one cycle.")
        if prefix_metadata.get("mechanical_formulation") != "reduced" or (
            extension_metadata.get("mechanical_formulation") != "reduced"
        ):
            raise ValueError("Both concatenated RHO seeds must use reduced mechanics.")

        prefix_keys = set(prefix_data.files) - {"metadata__json"}
        extension_keys = set(extension_data.files) - {"metadata__json"}
        if prefix_keys != extension_keys:
            raise ValueError("The RHO prefix and extension variables do not match.")

        payload: dict[str, np.ndarray] = {}
        maximum_boundary_change = 0.0
        for key in sorted(prefix_keys):
            prefix = np.asarray(prefix_data[key])
            extension = np.asarray(extension_data[key])
            if prefix.shape[:-1] != extension.shape[:-1]:
                raise ValueError(f"RHO seed '{key}' has incompatible row dimensions.")
            if key.startswith("states__"):
                intervals, remainder = divmod(prefix.shape[-1] - 1, prefix_cycles)
                if remainder or extension.shape[-1] != intervals + 1:
                    raise ValueError(f"RHO state '{key}' has incompatible cycle nodes.")
                stitched_prefix = prefix.copy()
                maximum_boundary_change = max(
                    maximum_boundary_change,
                    float(np.max(np.abs(stitched_prefix[..., -1] - extension[..., 0]))),
                )
                stitched_prefix[..., -1] = extension[..., 0]
                payload[key] = np.concatenate(
                    (stitched_prefix, extension[..., 1:]), axis=-1
                )
            elif key.startswith("controls__"):
                nodes, remainder = divmod(prefix.shape[-1], prefix_cycles)
                if remainder or extension.shape[-1] != nodes:
                    raise ValueError(
                        f"RHO control '{key}' has incompatible cycle nodes."
                    )
                payload[key] = np.concatenate((prefix, extension), axis=-1)
            else:
                if prefix.shape != extension.shape or not np.array_equal(
                    prefix, extension
                ):
                    raise ValueError(f"RHO auxiliary value '{key}' does not match.")
                payload[key] = prefix

    metadata = dict(prefix_metadata)
    metadata.update(
        {
            "cycles_per_window": prefix_cycles + 1,
            "producer_mode": "full_horizon_plus_terminal_rho",
            "producer_prefix_cycles": prefix_cycles,
            "producer_extension_cycles": 1,
            "replaced_reduced_boundary_maximum_absolute_change": (
                maximum_boundary_change
            ),
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def _benchmark_success(
    result_path: Path,
    *,
    expected_mode: str,
    expected_cycles: int,
    expected_solver: str,
    expected_mechanics: str | None = None,
) -> bool:
    """Require a solver and physical certificate for the complete horizon."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        if expected_mechanics is None:
            expected_mechanics = "full" if expected_mode == "single_shot" else "reduced"
        expected_use_sx = expected_solver == "ipopt" and expected_mode != "single_shot"
        return bool(
            result["success"]
            and result.get("solver_success") is True
            and result.get("physical_success") is True
            and result.get("solver") == expected_solver
            and result.get("mode") == expected_mode
            and int(result.get("covered_cycles") or 0) == expected_cycles
            and int(result.get("physically_validated_cycles") or 0) == expected_cycles
            and configuration.get("single_shot") is (expected_mode == "single_shot")
            and configuration.get("mechanical_formulation") == expected_mechanics
            and int(configuration.get("cycles_per_window") or 0)
            == (expected_cycles if expected_mode == "single_shot" else 1)
            and int(configuration.get("n_windows") or 0) == expected_cycles
            and configuration.get("use_sx") is expected_use_sx
            and configuration.get(f"{expected_solver}_linear_solver") == "mumps"
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _benchmark_payload_is_readable(result_path: Path) -> bool:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        return isinstance(result, dict) and result.get("error") is None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _rho_extension_success(result_path: Path) -> bool:
    """Certify one solved physical cycle independently of endurance semantics.

    A one-cycle extension is a warm-start construction, not an endurance
    campaign.  The global endurance verdict may therefore reject a physically
    validated cycle merely because a Ding capacity decreased.  The extension
    remains usable when IPOPT converged and its sole window is certified.
    """

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"]["ipopt"]
        windows = result.get("windows") or []
        return bool(
            result.get("solver_success") is True
            and result.get("solver") == "ipopt"
            and result.get("mode") == "rho"
            and int(result.get("covered_cycles") or 0) == 1
            and int(result.get("physically_validated_cycles") or 0) == 1
            and len(windows) == 1
            and windows[0].get("validated") is True
            and configuration.get("single_shot") is False
            and configuration.get("mechanical_formulation") == "reduced"
            and int(configuration.get("cycles_per_window") or 0) == 1
            and int(configuration.get("n_windows") or 0) == 1
            and configuration.get("use_sx") is True
            and configuration.get("ipopt_linear_solver") == "mumps"
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _log_has_unknown_mumps_warning(log_path: str | Path) -> bool:
    log_path = Path(log_path)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "libMAD WARNING: option linear_solver is of unknown type mumps" in text


def _benchmark_validated_cycles(
    result_path: Path,
    *,
    expected_mode: str,
    expected_solver: str,
    expected_requested_cycles: int,
) -> int:
    """Return the complete solver/physical prefix reported for one backend."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        if (
            result.get("solver") != expected_solver
            or result.get("mode") != expected_mode
            or configuration.get("single_shot") is not (expected_mode == "single_shot")
            or configuration.get("mechanical_formulation") != "reduced"
            or int(configuration.get("cycles_per_window") or 0) != 1
            or int(configuration.get("n_windows") or 0) != expected_requested_cycles
            or configuration.get("use_sx") is not True
            or configuration.get(f"{expected_solver}_linear_solver") != "mumps"
        ):
            return 0
        covered = int(result.get("covered_cycles") or 0)
        physical = int(result.get("physically_validated_cycles") or 0)
        return min(covered, physical)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 0


def _seed_cycle_count(seed_path: Path) -> int:
    try:
        with np.load(seed_path, allow_pickle=False) as data:
            return int(_load_metadata(data)["cycles_per_window"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _common_solver_options(args: argparse.Namespace) -> list[str]:
    return [
        "--objective",
        "fatigue",
        "--ipopt-profile",
        "periodic_collocation",
        "--ipopt-enforce-start-constraints",
        "--stimulations-per-cycle",
        "30",
        "--n-threads",
        str(args.n_threads),
        "--crank-assistance",
        str(args.crank_assistance),
        "--nlp-tolerance",
        "1e-8",
        "--primal-feasibility-threshold",
        "1e-5",
        "--standard-warmup-seed",
        str(
            args.workspace / ".github/benchmark-seeds/legacy-resistive-0p22-warmup.npz"
        ),
        "--legacy-standard-warmup-seed-signed-torque",
        "0.22",
        "--standard-warmup-seed-continuation",
        "--warmup-ipopt-linear-solver",
        "mumps",
        "--ipopt-linear-solver",
        "mumps",
        "--madnlp-linear-solver",
        "mumps",
        "--madnlp-max-iter",
        str(args.max_iterations),
        "--ipopt-disable-historical-initial-guess",
        "--reduced-cycling-profile",
        str(args.seed_dir / "reduced-cycling-fourier12.npz"),
        "--state-scaling",
        "full",
        "--first-node-wheel-q-slack",
        "0",
        "--terminal-wheel-q-slack",
        str(args.terminal_wheel_q_slack),
        "--compact-rho-output",
        "--print-traces",
    ]


def _rho_command(
    args: argparse.Namespace,
    result_path: Path,
    seed_path: Path,
    *,
    n_windows: int | None = None,
    common_initial_solution: Path | None = None,
) -> list[str]:
    if n_windows is None:
        n_windows = args.max_cycles
    if n_windows < 1:
        raise ValueError("n_windows must be strictly positive.")
    if common_initial_solution is None:
        common_initial_solution = args.seed_dir / "common-reduced.npz"
    return [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        "ipopt",
        *_common_solver_options(args),
        "--ipopt-use-sx",
        "--no-optional-nlp-periodic-ipopt-hot-start",
        "--ipopt-enable-periodic-fes-warmup-projection",
        "--periodic-fes-warmup-projection-strategy",
        "rollout",
        "--initial-guess-diagnostics",
        "--ipopt-max-iter",
        str(args.max_iterations),
        "--cycles-per-window",
        "1",
        "--n-windows",
        str(n_windows),
        "--max-consecutive-failing",
        "1",
        "--mechanical-formulation",
        "reduced",
        "--common-initial-solution",
        str(common_initial_solution),
        "--common-initial-solution-recenter-first-node-bounds",
        "--receding-horizon-solution-output",
        str(seed_path),
        "--allow-partial-receding-horizon-solution-output",
        "--output-json",
        str(result_path),
    ]


def _full_horizon_command(
    args: argparse.Namespace,
    cycles: int,
    seed_path: Path,
    result_path: Path,
    solution_path: Path,
    *,
    mechanical_formulation: str = "reduced",
    prefix_solution_path: Path | None = None,
) -> list[str]:
    if mechanical_formulation not in {"full", "reduced"}:
        raise ValueError("mechanical_formulation must be 'full' or 'reduced'.")
    full_horizon_solver = getattr(args, "full_horizon_solver", "madnlp")
    command = [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        full_horizon_solver,
        *_common_solver_options(args),
        "--ipopt-no-use-sx",
        "--ipopt-max-iter",
        str(args.max_iterations),
    ]
    if cycles >= 3:
        # The historical bridge has 60 controls and can initialize at most two
        # 30-stimulation cycles. Larger horizons consume the certified RHO
        # chronology directly instead of loading an incompatible warmup.
        command.extend(
            [
                "--ipopt-disable-standard-warmup",
                "--adopt-common-initial-solution-warmup-cycles",
            ]
        )
    command.extend(
        [
            "--optional-nlp-periodic-ipopt-hot-start",
            # Store the block/FES/RK4 seed defects for MadNLP as well as ACADOS.
            # These diagnostics distinguish a continuous RHO/FHO handoff from a
            # dynamically infeasible monolithic transcription.
            "--initial-guess-diagnostics",
            "--periodic-ipopt-refinement-use-sx",
            "--periodic-ipopt-refinement-iterations",
            str(args.max_iterations),
            "--single-shot",
            "--cycles-per-window",
            str(cycles),
            "--n-windows",
            str(cycles),
            "--mechanical-formulation",
            mechanical_formulation,
            "--full-contact-position-tolerance",
            "2e-5",
            "--common-initial-solution",
            str(seed_path),
            "--common-initial-solution-output",
            str(solution_path),
            "--output-json",
            str(result_path),
        ]
    )
    if full_horizon_solver == "madnlp":
        command.append("--exact-initial-nlp-audit")
    if prefix_solution_path is not None:
        command.extend(["--full-horizon-prefix-solution", str(prefix_solution_path)])
    return command


def _attempt_record(
    cycles: int,
    phase: str,
    monitored: MonitoredRun,
    result_path: Path,
    *,
    mechanical_formulation: str = "reduced",
    solution_path: Path | None = None,
    prefix_solution_path: Path | None = None,
    expected_solver: str = "madnlp",
) -> dict:
    unknown_mumps_warning = _log_has_unknown_mumps_warning(Path(monitored.log_path))
    certificate = _benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=cycles,
        expected_solver=expected_solver,
        expected_mechanics=mechanical_formulation,
    )
    solution_available = solution_path is None or solution_path.is_file()
    infrastructure_error = bool(
        not monitored.memory_limit_exceeded
        and not monitored.timed_out
        and (
            monitored.return_code != 0
            or not _benchmark_payload_is_readable(result_path)
            or unknown_mumps_warning
            or (certificate and not solution_available)
        )
    )
    success = bool(
        certificate
        and solution_available
        and monitored.return_code == 0
        and not monitored.memory_limit_exceeded
        and not monitored.timed_out
    )
    failure_kind = (
        None
        if success
        else (
            "memory_limit"
            if monitored.memory_limit_exceeded
            else (
                "timeout"
                if monitored.timed_out
                else (
                    "infrastructure_error" if infrastructure_error else "solver_failure"
                )
            )
        )
    )
    return {
        "cycles": cycles,
        "mechanical_formulation": mechanical_formulation,
        "phase": phase,
        "success": success,
        "failure_kind": failure_kind,
        "certificate_valid": certificate,
        "solution_available": solution_available,
        "infrastructure_error": infrastructure_error,
        "unknown_mumps_warning": unknown_mumps_warning,
        "result_path": str(result_path),
        "solution_path": None if solution_path is None else str(solution_path),
        "seed_origin": (
            "rho_plus_certified_fho_prefix"
            if prefix_solution_path is not None
            else "rho_prefix"
        ),
        "prefix_solution_path": (
            None if prefix_solution_path is None else str(prefix_solution_path)
        ),
        "peak_rss_bytes": monitored.peak_rss_bytes,
        "peak_rss_gib": monitored.peak_rss_bytes / GIB,
        "return_code": monitored.return_code,
        "elapsed_s": monitored.elapsed_s,
        "memory_limit_exceeded": monitored.memory_limit_exceeded,
        "timed_out": monitored.timed_out,
        "log_path": monitored.log_path,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# RHO reduced vs full-horizon size homotopy",
        "",
        f"- Limite RSS : `{report['rss_limit_gib']:.3f} GiB`",
        f"- Plafond demandé : `{report['max_cycles']} cycles`",
        f"- RHO de référence disponibles : `{report['rho_available_cycles']} cycles`",
        f"- Chaîne RHO/FHO construite : `{report.get('homotopy_constructed_cycles', 0)} cycles`",
        f"- Plus grand full horizon validé : `{report['largest_successful_cycles']}`",
        f"- Trous de convergence : `{report.get('solver_gap_cycles', [])}`",
        (
            "- Initialisation : `FHO_N certifié + un RHO résolu depuis son état "
            "terminal`"
        ),
        (
            "- Amorçage reduced à deux RHO : "
            f"`{'succès' if report['rho']['success'] else 'échec'}`, "
            f"pic RSS `{report['rho']['peak_rss_gib']:.3f} GiB`, "
            f"temps `{report['rho']['elapsed_s']:.1f} s`"
        ),
        f"- Arrêt : `{report['stop_reason']}`",
        "",
        "## Extensions RHO depuis le terminal FHO",
        "",
    ]
    extension_attempts = report.get("extension_rho_attempts", [])
    if extension_attempts:
        lines.extend(
            [
                "| Après FHO | RHO cible | Succès | Échec | Pic RSS (GiB) | Temps (s) |",
                "|---:|---:|:---:|:---|---:|---:|",
            ]
        )
        for extension in extension_attempts:
            lines.append(
                f"| {extension['after_full_horizon_cycles']} | "
                f"{extension['target_cycle']} | "
                f"{'oui' if extension['success'] else 'non'} | "
                f"{extension.get('failure_kind') or '—'} | "
                f"{extension['peak_rss_gib']:.3f} | "
                f"{extension['elapsed_s']:.1f} |"
            )
        lines.append("")
    else:
        lines.extend(["- Aucune extension nécessaire ou atteinte.", ""])
    lines.extend(
        [
            "## Sweep FHO reduced/MX",
            "",
            "| Cycles | Phase | Chance | Seed | Succès | Échec | Pic RSS (GiB) | Temps (s) |",
            "|---:|:---|---:|:---|:---:|:---|---:|---:|",
        ]
    )
    for attempt in report["full_horizon_attempts"]:
        lines.append(
            f"| {attempt['cycles']} | {attempt['phase']} | "
            f"{attempt.get('chance', 1)} | "
            f"{attempt.get('seed_origin', 'rho_prefix')} | "
            f"{'oui' if attempt['success'] else 'non'} | "
            f"{attempt.get('failure_kind') or '—'} | "
            f"{attempt['peak_rss_gib']:.3f} | {attempt['elapsed_s']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_memory_limit(raw: str, total_memory: int) -> float:
    if raw.lower() == "auto":
        return automatic_rss_limit_gib(total_memory)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--memory-limit-gib must be 'auto' or a positive number.")
    if value * GIB >= total_memory:
        raise ValueError(
            "--memory-limit-gib must leave headroom below the detected allocation."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--memory-limit-gib", default="auto")
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--full-horizon-solver",
        choices=("madnlp", "ipopt"),
        default="madnlp",
        help="NLP solver used for the reduced/MX monolithic FHO problems.",
    )
    parser.add_argument("--crank-assistance", type=float, default=0.0)
    parser.add_argument("--terminal-wheel-q-slack", type=float, default=0.002)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--attempt-timeout-s",
        type=float,
        default=1800.0,
        help="Wall-time cap for each independent solver attempt.",
    )
    return parser


def _run_horizon_attempt(
    args: argparse.Namespace,
    *,
    rho_seed: Path,
    cycles: int,
    phase: str,
    chance: int,
    rss_limit_bytes: int,
    mechanical_formulation: str = "reduced",
    prefix_solution_path: Path | None = None,
) -> dict:
    if chance < 1:
        raise ValueError("chance must be strictly positive.")
    case_prefix = "full-horizon"
    case_dir = args.output_dir / f"{case_prefix}-{cycles:04d}" / f"chance-{chance}"
    seed_path = case_dir / "rho-reduced-prefix.npz"
    result_path = case_dir / "result.json"
    solution_path = case_dir / "full-solution.npz"
    for stale_path in (result_path, solution_path):
        stale_path.unlink(missing_ok=True)
    write_rho_seed_prefix(rho_seed, seed_path, cycles)
    monitored = run_monitored(
        _full_horizon_command(
            args,
            cycles,
            seed_path,
            result_path,
            solution_path,
            mechanical_formulation=mechanical_formulation,
            prefix_solution_path=prefix_solution_path,
        ),
        cwd=args.workspace,
        log_path=case_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    return _attempt_record(
        cycles,
        phase,
        monitored,
        result_path,
        mechanical_formulation=mechanical_formulation,
        solution_path=solution_path,
        prefix_solution_path=prefix_solution_path,
        expected_solver=getattr(args, "full_horizon_solver", "madnlp"),
    )


def _run_extension_rho(
    args: argparse.Namespace,
    *,
    source_full_solution: Path,
    reference_rho_seed: Path,
    after_cycles: int,
    rss_limit_bytes: int,
) -> dict:
    """Homotope RHO_(N+1) from its RHO reference to the FHO_N terminal state."""

    case_dir = args.output_dir / f"rho-extension-after-fho-{after_cycles:04d}"
    reference_cycle_path = case_dir / "reference-rho-cycle.npz"
    write_rho_seed_cycle(reference_rho_seed, reference_cycle_path, after_cycles + 1)

    accepted_fraction = 0.0
    step = 0.25
    minimum_step = RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP
    source_solution_path = reference_cycle_path
    stages: list[dict] = []
    peak_rss_bytes = 0
    elapsed_s = 0.0
    final_solution_path: Path | None = None
    final_result_path: Path | None = None
    final_seed_path: Path | None = None
    failure_kind = None
    infrastructure_error = False
    unknown_mumps_warning = False
    memory_limit_exceeded = False
    timed_out = False
    return_code = 0

    while accepted_fraction < 1.0:
        fraction = min(1.0, accepted_fraction + step)
        attempt_number = len(stages) + 1
        stage_dir = case_dir / f"stage-{attempt_number:02d}-{fraction:.6f}"
        seed_path = stage_dir / "homotopy-seed.npz"
        result_path = stage_dir / "result.json"
        solution_path = stage_dir / "rho-solution.npz"
        for stale_path in (result_path, solution_path):
            stale_path.unlink(missing_ok=True)
        seed_metadata = write_rho_initial_state_homotopy_seed(
            source_solution_path,
            reference_cycle_path,
            source_full_solution,
            seed_path,
            fraction,
        )
        monitored = run_monitored(
            _rho_command(
                args,
                result_path,
                solution_path,
                n_windows=1,
                common_initial_solution=seed_path,
            ),
            cwd=args.workspace,
            log_path=stage_dir / "solver.log",
            rss_limit_bytes=rss_limit_bytes,
            poll_interval_s=args.poll_interval_s,
            timeout_s=args.attempt_timeout_s,
        )
        certificate = _rho_extension_success(result_path)
        solution_available = bool(
            solution_path.is_file() and _seed_cycle_count(solution_path) == 1
        )
        stage_unknown_warning = _log_has_unknown_mumps_warning(monitored.log_path)
        stage_infrastructure_error = bool(
            not monitored.memory_limit_exceeded
            and not monitored.timed_out
            and (
                monitored.return_code != 0
                or not _benchmark_payload_is_readable(result_path)
                or stage_unknown_warning
                or (certificate and not solution_available)
            )
        )
        stage_success = bool(
            certificate
            and solution_available
            and monitored.return_code == 0
            and not monitored.memory_limit_exceeded
            and not monitored.timed_out
        )
        stage_failure_kind = (
            None
            if stage_success
            else (
                "memory_limit"
                if monitored.memory_limit_exceeded
                else (
                    "timeout"
                    if monitored.timed_out
                    else (
                        "infrastructure_error"
                        if stage_infrastructure_error
                        else "solver_failure"
                    )
                )
            )
        )
        stages.append(
            {
                "attempt": attempt_number,
                "fraction": fraction,
                "step": step,
                "success": stage_success,
                "failure_kind": stage_failure_kind,
                "maximum_initial_state_change": seed_metadata[
                    "homotopy_maximum_initial_state_change"
                ],
                "result_path": str(result_path),
                "solution_path": str(solution_path),
                "seed_path": str(seed_path),
                "log_path": monitored.log_path,
                "peak_rss_gib": monitored.peak_rss_bytes / GIB,
                "elapsed_s": monitored.elapsed_s,
            }
        )
        peak_rss_bytes = max(peak_rss_bytes, monitored.peak_rss_bytes)
        elapsed_s += monitored.elapsed_s
        return_code = monitored.return_code
        unknown_mumps_warning = unknown_mumps_warning or stage_unknown_warning
        infrastructure_error = infrastructure_error or stage_infrastructure_error
        memory_limit_exceeded = memory_limit_exceeded or monitored.memory_limit_exceeded
        timed_out = timed_out or monitored.timed_out
        final_result_path = result_path
        final_seed_path = seed_path

        if stage_success:
            accepted_fraction = fraction
            source_solution_path = solution_path
            final_solution_path = solution_path
            step = min(0.25, step * 2.0)
            continue
        failure_kind = stage_failure_kind
        if (
            stage_infrastructure_error
            or monitored.memory_limit_exceeded
            or monitored.timed_out
            or step / 2.0 < minimum_step
        ):
            break
        step /= 2.0

    success = accepted_fraction == 1.0 and final_solution_path is not None
    if success:
        failure_kind = None
    return {
        "after_full_horizon_cycles": after_cycles,
        "target_cycle": after_cycles + 1,
        "success": success,
        "failure_kind": failure_kind,
        "certificate_valid": success,
        "solution_available": final_solution_path is not None,
        "infrastructure_error": infrastructure_error,
        "unknown_mumps_warning": unknown_mumps_warning,
        "seed_origin": "rho_reference_to_certified_fho_terminal_homotopy",
        "accepted_fraction": accepted_fraction,
        "homotopy_stages": stages,
        "reference_cycle_path": str(reference_cycle_path),
        "continuation_seed_path": (
            None if final_seed_path is None else str(final_seed_path)
        ),
        "continuation_source_cycles": after_cycles,
        "result_path": (None if final_result_path is None else str(final_result_path)),
        "solution_path": (
            None if final_solution_path is None else str(final_solution_path)
        ),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": peak_rss_bytes / GIB,
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "memory_limit_exceeded": memory_limit_exceeded,
        "timed_out": timed_out,
        "log_path": None if not stages else stages[-1]["log_path"],
    }


def run(args: argparse.Namespace) -> int:
    args.workspace = args.workspace.resolve()
    args.seed_dir = args.seed_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.max_cycles < 2:
        raise ValueError("--max-cycles must be at least two.")
    if args.n_threads < 1:
        raise ValueError("--n-threads must be strictly positive.")
    if args.poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be strictly positive.")
    if args.attempt_timeout_s <= 0:
        raise ValueError("--attempt-timeout-s must be strictly positive.")

    total_memory = available_memory_bytes()
    rss_limit_gib = _parse_memory_limit(args.memory_limit_gib, total_memory)
    rss_limit_bytes = int(rss_limit_gib * GIB)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "full-horizon-report.json"
    markdown_path = args.output_dir / "full-horizon-report.md"
    report = {
        "schema": "cocofest-full-horizon-sweep-v2",
        "max_cycles": args.max_cycles,
        "rho_available_cycles": 0,
        "total_memory_bytes": total_memory,
        "total_memory_gib": total_memory / GIB,
        "rss_limit_bytes": rss_limit_bytes,
        "rss_limit_gib": rss_limit_gib,
        "rho_graph": "SX",
        "full_horizon_graph": "MX",
        "rho_solver": "ipopt",
        "full_horizon_solver": args.full_horizon_solver,
        "linear_solver": "mumps",
        "initialization": "rho_reference_to_fho_terminal_state_homotopy",
        "rho": None,
        "paired_reduced_control_attempts": [],
        "extension_rho_attempts": [],
        "homotopy_constructed_cycles": 0,
        "full_horizon_attempts": [],
        "largest_successful_cycles": 0,
        "stop_reason": "not_started",
    }
    _write_report(report_path, report)

    rho_dir = args.output_dir / "rho-reduced"
    rho_result_path = rho_dir / "result.json"
    rho_seed_path = rho_dir / "concatenated-solution.npz"
    for stale_path in (rho_result_path, rho_seed_path):
        stale_path.unlink(missing_ok=True)
    rho_monitored = run_monitored(
        _rho_command(args, rho_result_path, rho_seed_path, n_windows=args.max_cycles),
        cwd=args.workspace,
        log_path=rho_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    report["rho"] = {
        **asdict(rho_monitored),
        "peak_rss_gib": rho_monitored.peak_rss_bytes / GIB,
        "result_path": str(rho_result_path),
        "seed_path": str(rho_seed_path),
    }
    rho_result_cycles = _benchmark_validated_cycles(
        rho_result_path,
        expected_mode="rho",
        expected_solver="ipopt",
        expected_requested_cycles=args.max_cycles,
    )
    rho_seed_cycles = _seed_cycle_count(rho_seed_path)
    rho_available_cycles = (
        rho_result_cycles if rho_result_cycles == rho_seed_cycles else 0
    )
    report["rho_available_cycles"] = rho_available_cycles
    report["rho"].update(
        {
            "success": (
                rho_available_cycles >= 2
                and rho_monitored.return_code == 0
                and not rho_monitored.memory_limit_exceeded
                and not rho_monitored.timed_out
                and not _log_has_unknown_mumps_warning(rho_monitored.log_path)
            ),
            "certificate_valid": rho_result_cycles >= 2,
            "unknown_mumps_warning": _log_has_unknown_mumps_warning(
                rho_monitored.log_path
            ),
            "validated_cycles": rho_result_cycles,
            "seed_cycles": rho_seed_cycles,
            "requested_ceiling_reached": rho_available_cycles == args.max_cycles,
        }
    )
    if not report["rho"]["success"]:
        rho_infrastructure_error = bool(
            not rho_monitored.memory_limit_exceeded
            and not rho_monitored.timed_out
            and (
                rho_monitored.return_code != 0
                or not _benchmark_payload_is_readable(rho_result_path)
            )
        )
        report["stop_reason"] = (
            "rho_memory_limit"
            if rho_monitored.memory_limit_exceeded
            else (
                "rho_timeout"
                if rho_monitored.timed_out
                else (
                    "rho_infrastructure_error"
                    if rho_infrastructure_error
                    else "rho_solver_failure"
                )
            )
        )
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        return 3 if rho_infrastructure_error else 2

    solver_gap_cycles: list[int] = []
    effective_max_cycles = min(args.max_cycles, rho_available_cycles)
    report["effective_max_cycles"] = effective_max_cycles
    current_seed_path = args.output_dir / "homotopy-seeds" / "rho-prefix-0002.npz"
    write_rho_seed_prefix(rho_seed_path, current_seed_path, 2)
    report["homotopy_constructed_cycles"] = 2
    current_full_solution: Path | None = None

    for cycles in horizon_sweep_targets(effective_max_cycles):
        if cycles > 2:
            if current_full_solution is None:
                raise RuntimeError(
                    "A certified FHO prefix is required for continuation."
                )
            extension_attempt = _run_extension_rho(
                args,
                source_full_solution=current_full_solution,
                reference_rho_seed=rho_seed_path,
                after_cycles=cycles - 1,
                rss_limit_bytes=rss_limit_bytes,
            )
            report["extension_rho_attempts"].append(extension_attempt)
            _write_report(report_path, report)
            if extension_attempt["infrastructure_error"]:
                report["stop_reason"] = "rho_extension_infrastructure_error"
                _write_report(report_path, report)
                _write_markdown(markdown_path, report)
                return 3
            if not extension_attempt["success"]:
                report["stop_reason"] = "rho_extension_" + str(
                    extension_attempt["failure_kind"]
                )
                break

            next_seed_path = (
                args.output_dir
                / "homotopy-seeds"
                / f"fho-{cycles - 1:04d}-plus-rho-{cycles:04d}.npz"
            )
            append_rho_extension_cycle(
                current_seed_path,
                Path(extension_attempt["solution_path"]),
                next_seed_path,
            )
            current_seed_path = next_seed_path
            report["homotopy_constructed_cycles"] = cycles

        attempt = _run_horizon_attempt(
            args,
            rho_seed=current_seed_path,
            cycles=cycles,
            phase="continuation",
            chance=1,
            rss_limit_bytes=rss_limit_bytes,
            prefix_solution_path=current_full_solution,
        )
        attempt["chance"] = 1
        report["full_horizon_attempts"].append(attempt)
        _write_report(report_path, report)
        if attempt["infrastructure_error"]:
            report["stop_reason"] = "infrastructure_error"
            _write_report(report_path, report)
            _write_markdown(markdown_path, report)
            return 3
        if not attempt["success"]:
            solver_gap_cycles.append(cycles)
            report["stop_reason"] = attempt["failure_kind"]
            break

        current_full_solution = Path(attempt["solution_path"])
        report["largest_successful_cycles"] = cycles
        report["stop_reason"] = (
            "requested_ceiling_reached"
            if cycles == args.max_cycles
            else (
                "rho_prefix_ceiling_reached"
                if cycles == effective_max_cycles
                else "running"
            )
        )

    report["solver_gap_cycles"] = solver_gap_cycles
    _write_markdown(markdown_path, report)
    _write_report(report_path, report)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
