#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_protocol_matched_sota_controls import DATASETS, constructors_for_dataset, slug


def write_status(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def complete(path: Path, kind: str, dataset: str, final_runs: int) -> bool:
    if kind == "sota":
        all_file, selected_file, expected_all, expected_selected = path / "sota_all_runs.csv", path / "sota_selected_runs.csv", 16 * final_runs, 4 * final_runs
    else:
        selected_groups = len(constructors_for_dataset(dataset)) * 3
        all_file = path / "constructor_all_runs.csv"
        selected_file = path / "constructor_selected_runs.csv"
        expected_all = selected_groups * 5 * final_runs
        expected_selected = selected_groups * final_runs
    try:
        return len(pd.read_csv(all_file)) == expected_all and len(pd.read_csv(selected_file)) == expected_selected
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False


def run_logged(command: list[str], log: Path, project: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\nCOMMAND: {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cora-output", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--global-output", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--final-runs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project = PROJECT_ROOT
    output_root, logs_dir, status_file = Path(args.output_root), Path(args.logs_dir), Path(args.status_file)
    output_root.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"state": "running", "phase": "sota", "datasets": {phase: {} for phase in ("sota", "constructors")}, "final_runs": args.final_runs}
    write_status(status_file, status)
    try:
        for phase in ("sota", "constructors"):
            status["phase"] = phase
            for dataset in DATASETS:
                target = output_root / slug(dataset) / phase
                phase_states = status["datasets"][phase]
                if complete(target, phase, dataset, args.final_runs):
                    phase_states[dataset] = "already_complete"
                    write_status(status_file, status)
                    continue
                phase_states[dataset] = "running"
                write_status(status_file, status)
                command = [sys.executable, "scripts/run_protocol_matched_sota_controls.py", "--data-root", args.data_root, "--cora-output", args.cora_output, "--protocol-root", args.protocol_root, "--output-root", str(output_root), "--phase", phase, "--datasets", dataset, "--final-runs", str(args.final_runs), "--device", args.device]
                run_logged(command, logs_dir / f"{phase}_{slug(dataset)}.log", project)
                if not complete(target, phase, dataset, args.final_runs):
                    raise RuntimeError(f"Incomplete output after success: {phase}/{dataset}")
                phase_states[dataset] = "complete"
                write_status(status_file, status)
        status["phase"] = "aggregation"
        write_status(status_file, status)
        run_logged([sys.executable, "scripts/aggregate_sota_controls.py", "--root", str(output_root), "--out", args.global_output, "--final-runs", str(args.final_runs)], logs_dir / "global_aggregation.log", project)
        status["state"], status["phase"] = "complete", "complete"
        write_status(status_file, status)
    except Exception as exc:
        status["state"], status["error"] = "failed", str(exc)
        write_status(status_file, status)
        raise


if __name__ == "__main__":
    main()
