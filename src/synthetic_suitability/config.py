"""Configuration and preregistered profiles for the synthetic study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class GraphDesignConfig:
    """Factorial graph-generation design.

    Each family exposes a small number of interpretable structural controls.
    Several sweeps hold node count and expected edge budget fixed so that the
    effect of bottlenecks, path length, or degree heterogeneity is not reduced
    to a density comparison.
    """

    node_counts: tuple[int, ...] = (240, 480)
    families: tuple[str, ...] = (
        "sbm_density_matched",
        "watts_strogatz",
        "barabasi_albert",
        "erdos_renyi",
        "degree_matched_bottleneck",
        "random_regular",
    )
    target_avg_degrees: tuple[int, ...] = (6, 12)
    community_counts: tuple[int, ...] = (2, 4)
    sbm_mixing: tuple[float, ...] = (0.03, 0.10, 0.25, 0.45)
    ws_rewire_probabilities: tuple[float, ...] = (0.0, 0.05, 0.20, 0.60, 1.0)
    ba_attachment: tuple[int, ...] = (2, 4, 8)
    bottleneck_edge_counts: tuple[int, ...] = (2, 4, 12, 32)
    latent_homophily_levels: tuple[float, ...] = (0.20, 0.50, 0.80)
    corruption_levels: tuple[float, ...] = (0.0, 0.25, 0.50)
    corruption_modes: tuple[str, ...] = ("degree_preserving", "homophily_attack")
    geometric_separations: tuple[float, ...] = (0.50, 1.50)
    degree_exponents: tuple[float, ...] = (2.20, 3.00)
    hierarchy_strengths: tuple[float, ...] = (2.00, 5.00)
    instances_per_configuration: int = 10
    ensure_connected: bool = False
    max_generation_attempts: int = 20


@dataclass(frozen=True)
class TaskDesignConfig:
    """Controlled node-classification tasks attached to every graph."""

    feature_regimes: tuple[str, ...] = ("informative", "partial", "noisy")
    feature_dimensions: int = 24
    class_signal: dict[str, float] = field(
        default_factory=lambda: {"informative": 1.50, "partial": 0.75, "noisy": 0.15}
    )
    feature_noise_std: float = 1.0
    label_regimes: tuple[str, ...] = ("structural",)
    task_seeds_per_graph: int = 3
    min_class_size: int = 12


@dataclass(frozen=True)
class PipelineStudyConfig:
    """Fixed pipeline settings shared by every synthetic condition."""

    pipeline_seeds_per_task: int = 3
    pipeline_seed_start: int = 41000
    rewiring_ratios: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40, 0.55)
    constructors: tuple[str, ...] = (
        "delaunay",
        "knn",
        "mutual_knn",
        "random_edge_budget_matched",
        "random_degree_matched",
    )
    refine_controls: bool = False
    downstream_backbone: str = "gcn"
    auxiliary_epochs: int = 200
    vgae_epochs: int = 400
    downstream_epochs: int = 400
    hidden_channels: int = 64
    latent_channels: int = 32
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    dropout: float = 0.35
    selection_metric: str = "f1"


@dataclass(frozen=True)
class MetricStudyConfig:
    """Pre-rewiring structural metric settings."""

    bfs_sources: int = 128
    resistance_pairs: int = 64
    exact_edge_connectivity_max_nodes: int = 500
    modularity_max_nodes: int = 3000
    spectral_tolerance: float = 1e-4


@dataclass(frozen=True)
class IndicatorStudyConfig:
    """Leakage-safe indicator fitting and evaluation settings."""

    success_thresholds_pp: tuple[float, ...] = (0.0, 0.5, 1.0)
    primary_success_threshold_pp: float = 0.5
    false_positive_cost: float = 3.0
    false_negative_cost: float = 1.0
    uncertain_margin: float = 0.12
    outer_grouping: str = "configuration_id"
    leave_one_family_out: bool = True
    bootstrap_repetitions: int = 1000
    random_state: int = 7301
    max_tree_depth: int = 3
    min_leaf_graphs: int = 12
    candidate_features: tuple[str, ...] = (
        "lambda2_norm_laplacian",
        "algebraic_connectivity",
        "density",
        "avg_degree",
        "degree_cv",
        "avg_clustering",
        "transitivity",
        "degree_assortativity",
        "approx_avg_shortest_path",
        "approx_diameter",
        "num_components",
        "largest_cc_ratio",
        "num_isolates",
        "effective_resistance_sample",
        "edge_connectivity",
        "edge_connectivity_ratio",
        "spectral_sweep_conductance",
        "modularity",
    )
    # These columns already exist for the 12 real graphs and therefore define
    # the feature boundary for the frozen externally validated indicator.
    external_compatible_features: tuple[str, ...] = (
        "lambda2_norm_laplacian",
        "density",
        "avg_degree",
        "degree_cv",
        "approx_avg_shortest_path",
        "approx_diameter",
        "num_components",
        "largest_cc_ratio",
        "num_isolates",
        "effective_resistance_sample",
    )


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Complete, serializable study configuration."""

    profile: str = "full"
    master_seed: int = 20260916
    output_dir: Path = Path("outputs_revision/synthetic_suitability")
    graph: GraphDesignConfig = field(default_factory=GraphDesignConfig)
    task: TaskDesignConfig = field(default_factory=TaskDesignConfig)
    pipeline: PipelineStudyConfig = field(default_factory=PipelineStudyConfig)
    metrics: MetricStudyConfig = field(default_factory=MetricStudyConfig)
    indicator: IndicatorStudyConfig = field(default_factory=IndicatorStudyConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SyntheticExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            profile=raw.get("profile", "custom"),
            master_seed=int(raw.get("master_seed", 20260916)),
            output_dir=Path(raw["output_dir"]),
            graph=GraphDesignConfig(**_tuplify(raw["graph"])),
            task=TaskDesignConfig(**_tuplify(raw["task"])),
            pipeline=PipelineStudyConfig(**_tuplify(raw["pipeline"])),
            metrics=MetricStudyConfig(**raw["metrics"]),
            indicator=IndicatorStudyConfig(**_tuplify(raw["indicator"])),
        )


def _tuplify(payload: dict[str, Any]) -> dict[str, Any]:
    converted = dict(payload)
    for key, value in list(converted.items()):
        if isinstance(value, list):
            converted[key] = tuple(value)
    return converted


def profile_config(name: str, output_dir: Path | None = None) -> SyntheticExperimentConfig:
    """Return a preregistered resource profile.

    ``full`` follows the paper protocol. ``pilot`` preserves the factorial
    logic with fewer repetitions. ``smoke`` is only for software validation
    and must never be used for scientific conclusions.
    """

    normalized = name.lower()
    base = SyntheticExperimentConfig(profile=normalized)
    if normalized == "full":
        config = base
    elif normalized == "pilot":
        config = replace(
            base,
            graph=replace(
                base.graph,
                node_counts=(240,),
                target_avg_degrees=(6, 12),
                community_counts=(2, 4),
                instances_per_configuration=5,
            ),
            task=replace(base.task, task_seeds_per_graph=2),
            pipeline=replace(
                base.pipeline,
                pipeline_seeds_per_task=2,
                auxiliary_epochs=100,
                vgae_epochs=150,
                downstream_epochs=200,
            ),
            indicator=replace(base.indicator, bootstrap_repetitions=500),
        )
    elif normalized == "smoke":
        config = replace(
            base,
            graph=replace(
                base.graph,
                node_counts=(90,),
                families=("sbm_density_matched", "watts_strogatz", "degree_matched_bottleneck"),
                target_avg_degrees=(6,),
                community_counts=(2,),
                sbm_mixing=(0.05, 0.35),
                ws_rewire_probabilities=(0.0, 0.6),
                bottleneck_edge_counts=(2, 12),
                instances_per_configuration=1,
            ),
            task=replace(
                base.task,
                feature_regimes=("partial",),
                feature_dimensions=12,
                task_seeds_per_graph=1,
                min_class_size=5,
            ),
            pipeline=replace(
                base.pipeline,
                pipeline_seeds_per_task=1,
                constructors=("delaunay", "knn", "random_edge_budget_matched"),
                rewiring_ratios=(0.0, 0.25),
                auxiliary_epochs=8,
                vgae_epochs=10,
                downstream_epochs=12,
                hidden_channels=16,
                latent_channels=8,
            ),
            metrics=replace(base.metrics, bfs_sources=32, resistance_pairs=12),
            indicator=replace(base.indicator, bootstrap_repetitions=50, min_leaf_graphs=2),
        )
    elif normalized in {"recovery_smoke", "recovery_pilot", "recovery_full"}:
        if normalized == "recovery_smoke":
            graph = replace(
                base.graph,
                node_counts=(90,), families=("latent_corruption_recovery",),
                target_avg_degrees=(6,), latent_homophily_levels=(0.50,),
                corruption_levels=(0.0, 0.50), instances_per_configuration=1,
            )
            task = replace(base.task, feature_regimes=("partial",), feature_dimensions=12,
                           task_seeds_per_graph=1, min_class_size=5)
            pipeline = replace(
                base.pipeline, pipeline_seeds_per_task=1,
                constructors=("delaunay", "raw_umap_delaunay", "knn", "mutual_knn", "mst_knn",
                              "learned_hd_knn", "raw_hd_knn", "delaunay_union_original",
                              "random_edge_budget_matched", "random_degree_matched"),
                rewiring_ratios=(0.0, 0.25), auxiliary_epochs=8, vgae_epochs=10,
                downstream_epochs=12, hidden_channels=16, latent_channels=8,
            )
            config = replace(base, graph=graph, task=task, pipeline=pipeline,
                             metrics=replace(base.metrics, bfs_sources=32, resistance_pairs=12),
                             indicator=replace(base.indicator, bootstrap_repetitions=50, min_leaf_graphs=2))
        else:
            full_scale = normalized == "recovery_full"
            graph = replace(
                base.graph,
                node_counts=(240, 600) if not full_scale else (240, 600, 1200),
                families=("latent_corruption_recovery",), target_avg_degrees=(6, 12),
                latent_homophily_levels=(0.20, 0.50, 0.80),
                corruption_levels=(0.0, 0.10, 0.25, 0.40, 0.60) if full_scale else (0.0, 0.25, 0.50),
                instances_per_configuration=8 if full_scale else 3,
            )
            task = replace(base.task, task_seeds_per_graph=3 if full_scale else 1)
            pipeline = replace(
                base.pipeline, pipeline_seeds_per_task=3 if full_scale else 2,
                constructors=("delaunay", "raw_umap_delaunay", "knn", "mutual_knn", "mst_knn",
                              "learned_hd_knn", "raw_hd_knn", "delaunay_union_original",
                              "random_edge_budget_matched", "random_degree_matched"),
            )
            config = replace(base, graph=graph, task=task, pipeline=pipeline)
    elif normalized in {"blind_wave1", "blind_wave2", "blind_wave3"}:
        family = {
            "blind_wave1": "latent_geometric_recovery",
            "blind_wave2": "latent_degree_corrected_recovery",
            "blind_wave3": "latent_hierarchical_recovery",
        }[normalized]
        graph = replace(
            base.graph,
            node_counts=(240, 600), families=(family,), target_avg_degrees=(6, 12),
            latent_homophily_levels=(0.20, 0.80), corruption_levels=(0.25, 0.50),
            corruption_modes=("degree_preserving", "homophily_attack"),
            geometric_separations=(0.50, 1.50), degree_exponents=(2.20, 3.00),
            hierarchy_strengths=(2.00, 5.00), instances_per_configuration=1,
        )
        # Keep each blind wave bounded while retaining repeated pipeline seeds.
        if normalized == "blind_wave2":
            graph = replace(graph, corruption_levels=(0.25,))
        if normalized == "blind_wave3":
            graph = replace(graph, corruption_modes=("degree_preserving",))
        task = replace(base.task, task_seeds_per_graph=1)
        pipeline = replace(
            base.pipeline, pipeline_seeds_per_task=2,
            constructors=("delaunay", "raw_umap_delaunay", "knn", "mutual_knn", "mst_knn",
                          "learned_hd_knn", "raw_hd_knn", "delaunay_union_original",
                          "random_edge_budget_matched", "random_degree_matched"),
        )
        config = replace(base, graph=graph, task=task, pipeline=pipeline)
    else:
        raise ValueError(
            f"Unknown profile: {name}. Expected smoke, pilot, full, recovery_smoke, "
            "recovery_pilot, recovery_full, blind_wave1, blind_wave2, or blind_wave3."
        )
    resolved_output = output_dir or Path(f"outputs_revision/synthetic_suitability_{normalized}")
    return replace(config, output_dir=resolved_output)
