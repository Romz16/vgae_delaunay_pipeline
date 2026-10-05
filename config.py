
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class SplitConfig:

    test_size: float = 0.20
    val_size_from_train_val: float = 0.25
    stratify: bool = True


@dataclass(frozen=True)
class OptunaConfig:

    vgae_trials: int = 30
    gnn_trials: int = 30
    timeout_seconds: int | None = None
    n_startup_trials: int = 8
    pruning_warmup_steps: int = 40


@dataclass(frozen=True)
class TrainingConfig:

    vgae_opt_epochs: int = 200
    vgae_final_epochs: int = 400
    gnn_opt_epochs: int = 200
    gnn_final_epochs: int = 400
    final_runs: int = 100
    verbose_every: int = 20
    step_size: int = 100
    step_gamma: float = 0.5
    selection_metric: str = "f1"
    validation_select_rewiring: bool = True


@dataclass(frozen=True)
class DelaunayConfig:

    pretrain_epochs: int = 200
    pretrain_hidden_channels: int = 64
    pretrain_lr: float = 0.01
    pretrain_weight_decay: float = 5e-4
    umap_components: int = 2
    umap_neighbors: int = 15
    umap_min_dist: float = 0.10
    umap_seed: int = 42
    trustworthiness_max_samples: int = 3000


@dataclass(frozen=True)
class RewiringConfig:

    strategy: str = "vgae"
    hybrid_alpha: float = 0.70
    ct_exact_max_nodes: int = 2000
    ct_num_eigenvectors: int = 128
    ct_candidate_multiplier: int = 20
    ct_min_search_factor: int = 3
    ct_eps: float = 1e-12


@dataclass(frozen=True)
class OutputConfig:

    root_dir: Path = Path("outputs")
    embeddings_dirname: str = "embeddings"
    results_dirname: str = "results"
    studies_dirname: str = "optuna_studies"
    revision_dirname: str = "outputs_revision"

    def embeddings_dir(self) -> Path:
        return self.root_dir / self.embeddings_dirname

    def results_dir(self) -> Path:
        return self.root_dir / self.results_dirname

    def studies_dir(self) -> Path:
        return self.root_dir / self.studies_dirname

    def revision_dir(self) -> Path:
        return self.root_dir / self.revision_dirname

    def figures_dir(self) -> Path:
        return self.revision_dir() / "figures"


@dataclass(frozen=True)
class PipelineConfig:

    dataset_name: str = "custom_graph"
    base_seed: int = 123
    run_seed_start: int = 12345
    rewiring_ratios: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.40, 0.55)
    backbones: Tuple[str, ...] = ("gcn", "gcn_residual", "sage", "gat")
    vgae_embedding_filename: str | None = None
    output_csv_filename: str | None = None
    output_md_filename: str | None = None
    add_self_loops_after_rewire: bool = True
    device: str = "auto"

    split: SplitConfig = field(default_factory=SplitConfig)
    rewiring: RewiringConfig = field(default_factory=RewiringConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    delaunay: DelaunayConfig = field(default_factory=DelaunayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def safe_dataset_name(self) -> str:
        return self.dataset_name.lower().replace(" ", "_").replace("-", "_")

    @property
    def embedding_path(self) -> Path:
        filename = self.vgae_embedding_filename or f"{self.safe_dataset_name}_vgae_embeddings.npy"
        return self.output.embeddings_dir() / filename

    @property
    def result_csv_path(self) -> Path:
        filename = self.output_csv_filename or f"tabela_resultados_{self.safe_dataset_name}_100runs.csv"
        return self.output.results_dir() / filename

    @property
    def result_md_path(self) -> Path:
        filename = self.output_md_filename or f"tabela_resultados_{self.safe_dataset_name}_100runs.md"
        return self.output.results_dir() / filename


DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, object]] = {
    "hidden_channels": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "n_layers": {"type": "int", "low": 2, "high": 4},
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
    "dropout": {"type": "float", "low": 0.0, "high": 0.7},
    "activation": {"type": "categorical", "choices": ["relu", "leaky_relu", "elu", "gelu"]},
    "batch_norm": {"type": "categorical", "choices": [True, False]},
    "scheduler": {"type": "categorical", "choices": ["none", "step", "cosine"]},
    "gat_heads": {"type": "categorical", "choices": [2, 4, 8]},
}
