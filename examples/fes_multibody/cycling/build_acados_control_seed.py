#!/usr/bin/env python3
"""Overlay one certified ACADOS control cycle on a solver-neutral state seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


STATE_PREFIX = "states__"
CONTROL_PREFIX = "controls__"
METADATA_KEY = "metadata__json"
PW_MINIMUM_S = 131.405e-6
PW_MAXIMUM_S = 600e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in archive.files
            if key != METADATA_KEY
        }
        metadata = (
            json.loads(str(archive[METADATA_KEY].item()))
            if METADATA_KEY in archive.files
            else None
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"Seed '{path}' has no JSON metadata.")
    return arrays, metadata


def _positive_integer(metadata: dict, key: str, path: Path) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"Seed '{path}' has invalid {key}={value!r}.")
    return int(value)


def build_hybrid_control_seed(
    common_seed_path: Path,
    certified_acados_path: Path,
    output_path: Path,
    *,
    cycle: int,
) -> dict:
    """Keep every common state and replace controls by one ACADOS cycle."""

    common_seed_path = Path(common_seed_path).resolve()
    certified_acados_path = Path(certified_acados_path).resolve()
    output_path = Path(output_path).resolve()
    common, common_metadata = _load_archive(common_seed_path)
    certified, certified_metadata = _load_archive(certified_acados_path)

    common_cycles = _positive_integer(
        common_metadata, "cycles_per_window", common_seed_path
    )
    if common_cycles != 1:
        raise ValueError("The common target seed must contain exactly one cycle.")
    source_cycles = _positive_integer(
        certified_metadata, "cycles_per_window", certified_acados_path
    )
    stimulations = _positive_integer(
        common_metadata, "stimulations_per_cycle", common_seed_path
    )
    source_stimulations = _positive_integer(
        certified_metadata, "stimulations_per_cycle", certified_acados_path
    )
    if source_stimulations != stimulations:
        raise ValueError(
            "The common and ACADOS seeds use different stimulation counts "
            f"({stimulations} versus {source_stimulations})."
        )
    for key in (
        "schema",
        "model_formulation",
        "mechanical_formulation",
        "constant_crank_torque",
        "torque_application",
        "calcium_forcing_formulation",
        "ding_sum_stim_truncation",
        "activate_force_length_relationship",
        "activate_force_velocity_relationship",
        "activate_passive_force_relationship",
        "pulse_width_minimum_policy",
        "pulse_width_maximum_s",
    ):
        if common_metadata.get(key) != certified_metadata.get(key):
            raise ValueError(
                f"The control seed has {key}={certified_metadata.get(key)!r}, "
                f"expected {common_metadata.get(key)!r}."
            )
    if cycle < 1 or cycle > source_cycles:
        raise ValueError(
            f"Requested ACADOS control cycle {cycle}; available range is "
            f"1..{source_cycles}."
        )

    common_control_keys = sorted(
        key for key in common if key.startswith(CONTROL_PREFIX)
    )
    if not common_control_keys:
        raise ValueError("The common seed has no controls.")
    start = (cycle - 1) * stimulations
    stop = cycle * stimulations
    control_summary = {}
    for key in common_control_keys:
        if key not in certified:
            raise KeyError(f"Certified ACADOS trajectory is missing '{key}'.")
        target = np.asarray(common[key], dtype=float)
        source = np.asarray(certified[key], dtype=float)
        if target.ndim != 2 or target.shape[1] != stimulations:
            raise ValueError(
                f"Common control '{key}' has shape {target.shape}; expected "
                f"(*, {stimulations})."
            )
        if source.ndim != 2 or source.shape[1] != source_cycles * stimulations:
            raise ValueError(
                f"Certified control '{key}' has shape {source.shape}; expected "
                f"(*, {source_cycles * stimulations})."
            )
        selected = source[:, start:stop]
        if selected.shape != target.shape or not np.all(np.isfinite(selected)):
            raise ValueError(
                f"Selected control '{key}' has invalid shape or non-finite values."
            )
        if (
            np.min(selected) < PW_MINIMUM_S - 1e-12
            or np.max(selected) > PW_MAXIMUM_S + 1e-12
        ):
            raise ValueError(
                f"Selected pulse widths for '{key}' leave "
                f"[{PW_MINIMUM_S}, {PW_MAXIMUM_S}] s."
            )
        common[key] = selected.copy()
        control_summary[key.removeprefix(CONTROL_PREFIX)] = {
            "minimum_s": float(np.min(selected)),
            "mean_s": float(np.mean(selected)),
            "maximum_s": float(np.max(selected)),
            "maximum_change_from_common_s": float(np.max(np.abs(selected - target))),
        }

    state_keys = [key for key in common if key.startswith(STATE_PREFIX)]
    if not state_keys:
        raise ValueError("The common seed has no states.")
    metadata = dict(common_metadata)
    metadata["initial_control_seed"] = {
        "kind": "certified_acados_rho_cycle",
        "source_sha256": _sha256(certified_acados_path),
        "source_cycle": int(cycle),
        "source_cycles": int(source_cycles),
        "producer_solver": certified_metadata.get("producer_solver"),
        "producer_transcription_profile": certified_metadata.get(
            "producer_transcription_profile"
        ),
        "state_source_sha256": _sha256(common_seed_path),
    }
    common[METADATA_KEY] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{output_path.stem}-",
        suffix=".npz",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez(temporary_path, **common)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "output": str(output_path),
        "cycle": int(cycle),
        "state_count": len(state_keys),
        "control_count": len(common_control_keys),
        "controls": control_summary,
        "metadata": metadata["initial_control_seed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-seed", type=Path, required=True)
    parser.add_argument("--certified-acados-trajectory", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = build_hybrid_control_seed(
        args.common_seed,
        args.certified_acados_trajectory,
        args.output,
        cycle=args.cycle,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
