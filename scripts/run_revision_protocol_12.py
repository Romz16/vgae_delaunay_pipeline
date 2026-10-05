#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DATASETS = [
    "Coauthor-CS",
    "Pubmed",
    "Airports-USA",
    "Airports-Brazil",
    "Airports-Europe",
    "Amazon-Photo",
    "Cornell",
    "Actor",
    "Roman-Empire",
    "Texas",
    "Minesweeper",
]


def slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def raw_results_complete(output_dir: Path) -> bool:
    path = output_dir / "outputs_revision" / "raw_feature_delaunay_results.csv"
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 3 and all(row.get("Protocol") == "raw_features_umap_delaunay" for row in rows)


def dataset_complete(output_dir: Path) -> bool:
    selected = output_dir / "outputs_revision" / "selected_rates.csv"
    final = output_dir / "outputs_revision" / "final_selected_test_results.csv"
    trust = output_dir / "outputs_revision" / "umap_trustworthiness.csv"
    return all(path.exists() and path.stat().st_size > 0 for path in [selected, final, trust]) and raw_results_complete(output_dir)


def write_status(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, project: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cora-output", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--global-output", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    cora_output = Path(args.cora_output)
    output_root = Path(args.output_root)
    global_output = Path(args.global_output)
    logs_dir = Path(args.logs_dir)
    status_file = Path(args.status_file)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"state": "running", "datasets": {}, "cora_backfill": "pending"}
    write_status(status_file, status)

    try:
        if raw_results_complete(cora_output):
            status["cora_backfill"] = "already_complete"
        else:
            status["cora_backfill"] = "running"
            write_status(status_file, status)
            run_logged(
                [
                    sys.executable,
                    "scripts/run_raw_feature_ablation.py",
                    "--dataset",
                    "Cora",
                    "--root",
                    str(data_root),
                    "--output-dir",
                    str(cora_output),
                ],
                logs_dir / "cora_raw_feature_backfill.log",
                project,
            )
            status["cora_backfill"] = "complete"
            write_status(status_file, status)

        for dataset in DATASETS:
            output_dir = output_root / slug(dataset)
            dataset_states = status["datasets"]
            assert isinstance(dataset_states, dict)
            if dataset_complete(output_dir):
                dataset_states[dataset] = "already_complete"
                write_status(status_file, status)
                continue
            if output_dir.exists() and any(output_dir.iterdir()):
                raise RuntimeError(f"Partial output exists and will not be overwritten automatically: {output_dir}")
            dataset_states[dataset] = "running"
            write_status(status_file, status)
            command = [
                sys.executable,
                "main.py",
                "--dataset",
                dataset,
                "--root",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--selection-metric",
                "f1",
                "--ratios",
                "0",
                "0.10",
                "0.25",
                "0.40",
                "0.55",
                "--backbones",
                "gcn",
                "sage",
                "gat",
            ]
            run_logged(command, logs_dir / f"{slug(dataset)}.log", project)
            if not dataset_complete(output_dir):
                raise RuntimeError(f"Dataset command returned success but required artifacts are incomplete: {dataset}")
            dataset_states[dataset] = "complete"
            write_status(status_file, status)

        status["state"] = "aggregating"
        write_status(status_file, status)
        run_logged(
            [
                sys.executable,
                "scripts/aggregate_revision_outputs.py",
                "--root",
                str(cora_output),
                str(output_root),
                "--out",
                str(global_output),
            ],
            logs_dir / "global_aggregation.log",
            project,
        )
        status["state"] = "complete"
        write_status(status_file, status)
        (status_file.parent / "protocol_12_complete.txt").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        write_status(status_file, status)
        raise


if __name__ == "__main__":
    main()
