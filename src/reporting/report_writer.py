"""Result consolidation and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.metrics import format_metric


class ReportWriter:
    """Create CSV and Markdown reports in the experiment format."""

    @staticmethod
    def make_row(
        modelo: str,
        rewiring: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one legacy result row for one graph condition."""
        acc = result["acc_summary"]
        f1 = result["f1_summary"]
        loss = result["loss_summary"]
        return {
            "Modelo": modelo,
            "Rewiring": rewiring,
            "acc_mean": acc["mean"],
            "acc_std": acc["std"],
            "acc_ci95": acc["ci95"],
            "acc_ci95_lower": acc.get("ci95_lower"),
            "acc_ci95_upper": acc.get("ci95_upper"),
            "f1_mean": f1["mean"],
            "f1_std": f1["std"],
            "f1_ci95": f1["ci95"],
            "f1_ci95_lower": f1.get("ci95_lower"),
            "f1_ci95_upper": f1.get("ci95_upper"),
            "loss_mean": loss["mean"],
            "loss_std": loss["std"],
            "loss_ci95": loss["ci95"],
            "Test_Acc": format_metric(acc["mean"], acc["std"], acc["ci95"]),
            "Test_F1_Macro": format_metric(f1["mean"], f1["std"], f1["ci95"]),
            "Train_Loss": format_metric(loss["mean"], loss["std"], loss["ci95"]),
            "elapsed_seconds": result["elapsed_seconds"],
        }

    @staticmethod
    def make_selected_summary_row(modelo: str, result: dict[str, Any]) -> dict[str, Any]:
        """Build official validation-selected summary row."""
        baseline_acc = result["baseline_acc_summary"]
        baseline_f1 = result["baseline_f1_summary"]
        selected_acc = result["selected_acc_summary"]
        selected_f1 = result["selected_f1_summary"]
        gain_acc = result["gain_acc_summary"]
        gain_f1 = result["gain_f1_summary"]
        loss = result["loss_summary"]
        return {
            "Modelo": modelo,
            "Protocol": "validation_selected_rewiring",
            "baseline_acc_mean": baseline_acc["mean"],
            "baseline_f1_mean": baseline_f1["mean"],
            "selected_acc_mean": selected_acc["mean"],
            "selected_f1_mean": selected_f1["mean"],
            "gain_acc_mean": gain_acc["mean"],
            "gain_acc_std": gain_acc["std"],
            "gain_acc_ci95": gain_acc["ci95"],
            "gain_acc_ci95_lower": gain_acc.get("ci95_lower"),
            "gain_acc_ci95_upper": gain_acc.get("ci95_upper"),
            "gain_f1_mean": gain_f1["mean"],
            "gain_f1_std": gain_f1["std"],
            "gain_f1_ci95": gain_f1["ci95"],
            "gain_f1_ci95_lower": gain_f1.get("ci95_lower"),
            "gain_f1_ci95_upper": gain_f1.get("ci95_upper"),
            "loss_mean": loss["mean"],
            "elapsed_seconds": result["elapsed_seconds"],
            "Selected_Test_Acc": format_metric(selected_acc["mean"], selected_acc["std"], selected_acc["ci95"]),
            "Selected_Test_F1_Macro": format_metric(selected_f1["mean"], selected_f1["std"], selected_f1["ci95"]),
            "Gain_F1_Macro": format_metric(gain_f1["mean"], gain_f1["std"], gain_f1["ci95"]),
        }

    @staticmethod
    def export(rows: list[dict[str, Any]], csv_path: Path, md_path: Path | None = None) -> pd.DataFrame:
        """Export result rows to CSV and optionally Markdown."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        if md_path is not None:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_markdown(md_path, index=False)
        return df
