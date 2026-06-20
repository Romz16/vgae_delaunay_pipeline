"""CLI entry point for the VGAE + Delaunay rewiring pipeline.

Examples:
    Run with a PyG dataset import:
        python main.py --dataset Cora --root ./data --output-dir outputs/cora

    Run with a local PyG .pt graph:
        python main.py --graph-file ./my_graph.pt --dataset MyGraph --output-dir outputs/my_graph

    Faster smoke test:
        python main.py --dataset Wisconsin --vgae-trials 3 --gnn-trials 3 \
            --final-runs 2 --gnn-opt-epochs 20 --gnn-final-epochs 20 \
            --vgae-opt-epochs 20 --vgae-final-epochs 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import (
    DelaunayConfig,
    OptunaConfig,
    OutputConfig,
    PipelineConfig,
    SplitConfig,
    TrainingConfig,
)
from src.data.loaders import GraphLoader
from src.pipeline.orchestrator import VGAEDelaunayPipeline
from src.utils.device import resolve_device
from src.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Pipeline VGAE + DGlf/Delaunay Rewiring"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset", type=str, help="Nome de dataset suportado pelo loader PyG."
    )
    input_group.add_argument(
        "--graph-file", type=str, help="Caminho para arquivo de grafo local."
    )

    parser.add_argument(
        "--graph-format",
        type=str,
        default=None,
        help="Formato explícito do arquivo de grafo.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Nome usado em arquivos de saída.",
    )
    parser.add_argument(
        "--root", type=str, default="./data", help="Diretório raiz dos datasets PyG."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Diretório de saída."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="auto, cpu, cuda ou cuda:0."
    )

    parser.add_argument(
        "--vgae-trials", type=int, default=30, help="Trials Optuna para VGAE."
    )
    parser.add_argument(
        "--gnn-trials", type=int, default=30, help="Trials Optuna por backbone GNN."
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=None, help="Timeout Optuna por estudo."
    )

    parser.add_argument(
        "--vgae-opt-epochs", type=int, default=200, help="Épocas por trial do VGAE."
    )
    parser.add_argument(
        "--vgae-final-epochs", type=int, default=400, help="Épocas do VGAE final."
    )
    parser.add_argument(
        "--gnn-opt-epochs", type=int, default=200, help="Épocas por trial das GNNs."
    )
    parser.add_argument(
        "--gnn-final-epochs",
        type=int,
        default=400,
        help="Épocas por run na avaliação final.",
    )
    parser.add_argument(
        "--final-runs", type=int, default=100, help="Número de runs finais."
    )

    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.25, 0.40, 0.55],
        help="Taxas de rewiring. Ex: --ratios 0 0.1 0.25 0.4 0.55",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=["gcn", "gcn_residual", "sage", "gat"],
        help="Backbones a avaliar.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed base.")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR."
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:

    dataset_name = args.dataset_name or args.dataset or Path(args.graph_file).stem
    output = OutputConfig(root_dir=Path(args.output_dir))
    optuna = OptunaConfig(
        vgae_trials=args.vgae_trials,
        gnn_trials=args.gnn_trials,
        timeout_seconds=args.timeout_seconds,
    )
    training = TrainingConfig(
        vgae_opt_epochs=args.vgae_opt_epochs,
        vgae_final_epochs=args.vgae_final_epochs,
        gnn_opt_epochs=args.gnn_opt_epochs,
        gnn_final_epochs=args.gnn_final_epochs,
        final_runs=args.final_runs,
    )
    return PipelineConfig(
        dataset_name=dataset_name,
        base_seed=args.seed,
        rewiring_ratios=tuple(args.ratios),
        backbones=tuple(args.backbones),
        output=output,
        optuna=optuna,
        training=training,
        split=SplitConfig(),
        delaunay=DelaunayConfig(),
        device=args.device,
    )


def main() -> None:
    """Load a graph and run the full pipeline."""
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = configure_logging(level)
    config = build_config(args)
    device = resolve_device(config.device)
    logger.info("Dispositivo selecionado: %s", device)

    loader = GraphLoader(root=args.root, device=device)
    if args.dataset:
        data = loader.load_from_name(args.dataset)
    else:
        data = loader.load_from_file(args.graph_file, args.graph_format)

    pipeline = VGAEDelaunayPipeline(config=config, device=device)
    pipeline.run(data)


if __name__ == "__main__":
    main()
