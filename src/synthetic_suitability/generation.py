
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable
import hashlib
import json

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from .config import GraphDesignConfig, TaskDesignConfig


@dataclass(frozen=True)
class GraphSpecification:
    family: str
    configuration_id: str
    parameters: dict[str, object]


@dataclass
class SyntheticGraph:
    graph_id: str
    specification: GraphSpecification
    instance_index: int
    generation_seed: int
    graph: nx.Graph
    structural_labels: np.ndarray
    generation_metadata: dict[str, object]


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    graph_id: str
    feature_regime: str
    task_seed: int
    features: np.ndarray
    labels: np.ndarray


def stable_seed(*parts: object, modulus: int = 2**31 - 1) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulus


def configuration_grid(cfg: GraphDesignConfig) -> list[GraphSpecification]:
    specs: list[GraphSpecification] = []
    for n, degree in product(cfg.node_counts, cfg.target_avg_degrees):
        if "sbm_density_matched" in cfg.families:
            for communities, mixing in product(cfg.community_counts, cfg.sbm_mixing):
                if n % communities != 0:
                    continue
                params = {"n": n, "target_avg_degree": degree, "communities": communities, "mixing": mixing}
                specs.append(_spec("sbm_density_matched", params))
        if "watts_strogatz" in cfg.families:
            k = _even_degree(degree, n)
            for beta in cfg.ws_rewire_probabilities:
                specs.append(_spec("watts_strogatz", {"n": n, "k": k, "beta": beta}))
        if "barabasi_albert" in cfg.families:
            for m in cfg.ba_attachment:
                if m < n:
                    specs.append(_spec("barabasi_albert", {"n": n, "m": m}))
        if "erdos_renyi" in cfg.families:
            p = min(0.95, degree / max(n - 1, 1))
            specs.append(_spec("erdos_renyi", {"n": n, "p": p, "target_avg_degree": degree}))
        if "degree_matched_bottleneck" in cfg.families and n % 2 == 0:
            d = _regular_degree(degree, n // 2)
            for bridges in cfg.bottleneck_edge_counts:
                max_bridges = (n * d) // 4
                if bridges <= max_bridges:
                    specs.append(_spec("degree_matched_bottleneck", {"n": n, "degree": d, "bridges": bridges}))
        if "random_regular" in cfg.families:
            d = _regular_degree(degree, n)
            specs.append(_spec("random_regular", {"n": n, "degree": d}))
        if "latent_corruption_recovery" in cfg.families and n % 2 == 0:
            for homophily in cfg.latent_homophily_levels:
                for corruption in cfg.corruption_levels:
                    modes = ("none",) if float(corruption) == 0.0 else cfg.corruption_modes
                    for mode in modes:
                        specs.append(_spec("latent_corruption_recovery", {
                            "n": n, "target_avg_degree": degree,
                            "latent_homophily": float(homophily),
                            "corruption_fraction": float(corruption),
                            "corruption_mode": mode,
                        }))
        if "latent_geometric_recovery" in cfg.families and n % 2 == 0:
            for separation, corruption, mode in product(
                cfg.geometric_separations, cfg.corruption_levels, cfg.corruption_modes
            ):
                specs.append(_spec("latent_geometric_recovery", {
                    "n": n, "target_avg_degree": degree, "geometric_separation": float(separation),
                    "corruption_fraction": float(corruption), "corruption_mode": mode,
                }))
        if "latent_degree_corrected_recovery" in cfg.families and n % 2 == 0:
            for homophily, exponent, corruption, mode in product(
                cfg.latent_homophily_levels, cfg.degree_exponents, cfg.corruption_levels, cfg.corruption_modes
            ):
                specs.append(_spec("latent_degree_corrected_recovery", {
                    "n": n, "target_avg_degree": degree, "latent_homophily": float(homophily),
                    "degree_exponent": float(exponent), "corruption_fraction": float(corruption),
                    "corruption_mode": mode,
                }))
        if "latent_hierarchical_recovery" in cfg.families and n % 4 == 0:
            for homophily, strength, corruption, mode in product(
                cfg.latent_homophily_levels, cfg.hierarchy_strengths, cfg.corruption_levels, cfg.corruption_modes
            ):
                specs.append(_spec("latent_hierarchical_recovery", {
                    "n": n, "target_avg_degree": degree, "latent_homophily": float(homophily),
                    "hierarchy_strength": float(strength), "corruption_fraction": float(corruption),
                    "corruption_mode": mode,
                }))
    # Some families do not use every outer design factor (for example BA does
    # not use target_avg_degree directly). Deduplicate their canonical specs so
    # graph ids and resumable part files remain one-to-one with configurations.
    return list({spec.configuration_id: spec for spec in specs}.values())


def _spec(family: str, parameters: dict[str, object]) -> GraphSpecification:
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    return GraphSpecification(family=family, configuration_id=f"{family}__{suffix}", parameters=parameters)


def _even_degree(degree: int, n: int) -> int:
    k = min(int(degree), n - 1)
    return k if k % 2 == 0 else max(2, k - 1)


def _regular_degree(degree: int, n: int) -> int:
    d = min(int(degree), n - 1)
    if (n * d) % 2:
        d -= 1
    return max(2, d)


def generate_graphs(cfg: GraphDesignConfig, master_seed: int) -> Iterable[SyntheticGraph]:
    for spec in configuration_grid(cfg):
        for instance in range(cfg.instances_per_configuration):
            seed = stable_seed(master_seed, spec.configuration_id, instance)
            graph, labels, metadata = generate_one(spec, seed, cfg)
            graph_id = f"{spec.configuration_id}__i{instance:02d}"
            yield SyntheticGraph(graph_id, spec, instance, seed, graph, labels, metadata)


def generate_one(
    spec: GraphSpecification,
    seed: int,
    cfg: GraphDesignConfig,
) -> tuple[nx.Graph, np.ndarray, dict[str, object]]:
    last_graph: nx.Graph | None = None
    for attempt in range(cfg.max_generation_attempts):
        attempt_seed = stable_seed(seed, attempt)
        family = spec.family
        p = spec.parameters
        if family == "sbm_density_matched":
            graph, labels, metadata = _sbm_density_matched(p, attempt_seed)
        elif family == "watts_strogatz":
            graph = nx.watts_strogatz_graph(int(p["n"]), int(p["k"]), float(p["beta"]), seed=attempt_seed)
            labels = _spectral_labels(graph, 2, attempt_seed)
            metadata = {"controlled_property": "rewiring_probability", "density_control": "fixed_n_and_k"}
        elif family == "barabasi_albert":
            graph = nx.barabasi_albert_graph(int(p["n"]), int(p["m"]), seed=attempt_seed)
            labels = _spectral_labels(graph, 2, attempt_seed)
            metadata = {"controlled_property": "degree_heterogeneity", "edge_budget": graph.number_of_edges()}
        elif family == "erdos_renyi":
            graph = nx.gnp_random_graph(int(p["n"]), float(p["p"]), seed=attempt_seed)
            labels = _spectral_labels(graph, 2, attempt_seed)
            metadata = {"controlled_property": "simple_random_connectivity"}
        elif family == "degree_matched_bottleneck":
            graph, labels, metadata = _degree_matched_bottleneck(p, attempt_seed)
        elif family == "random_regular":
            graph = nx.random_regular_graph(int(p["degree"]), int(p["n"]), seed=attempt_seed)
            labels = _spectral_labels(graph, 2, attempt_seed)
            metadata = {"controlled_property": "homogeneous_degree_control"}
        elif family == "latent_corruption_recovery":
            graph, labels, metadata = _latent_corruption_recovery(p, attempt_seed)
        elif family == "latent_geometric_recovery":
            graph, labels, metadata = _latent_geometric_recovery(p, attempt_seed)
        elif family == "latent_degree_corrected_recovery":
            graph, labels, metadata = _latent_degree_corrected_recovery(p, attempt_seed)
        elif family == "latent_hierarchical_recovery":
            graph, labels, metadata = _latent_hierarchical_recovery(p, attempt_seed)
        else:
            raise ValueError(f"Unknown family: {family}")
        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.add_nodes_from(range(int(p["n"])))
        last_graph = graph
        if not cfg.ensure_connected or nx.is_connected(graph):
            return graph, _ensure_viable_labels(graph, labels, seed), {**metadata, "generation_attempt": attempt}
    if last_graph is None:
        raise RuntimeError(f"Could not generate {spec.configuration_id}")
    return last_graph, _ensure_viable_labels(last_graph, labels, seed), {**metadata, "generation_attempt": cfg.max_generation_attempts - 1, "connectivity_requirement_failed": True}


def _sbm_density_matched(p: dict[str, object], seed: int):
    n = int(p["n"]); c = int(p["communities"]); d = float(p["target_avg_degree"]); mixing = float(p["mixing"])
    size = n // c
    external_degree = d * mixing
    internal_degree = max(0.1, d - external_degree)
    p_in = min(0.98, internal_degree / max(size - 1, 1))
    p_out = min(0.98, external_degree / max(n - size, 1))
    probs = np.full((c, c), p_out, dtype=float)
    np.fill_diagonal(probs, p_in)
    graph = nx.stochastic_block_model([size] * c, probs.tolist(), seed=seed)
    labels = np.repeat(np.arange(c, dtype=np.int64), size)
    return graph, labels, {
        "controlled_property": "community_mixing_at_fixed_expected_degree",
        "p_in": p_in, "p_out": p_out, "expected_degree": d,
    }


def _latent_corruption_recovery(p: dict[str, object], seed: int):
    n = int(p["n"]); target_degree = float(p["target_avg_degree"])
    target_h = float(p["latent_homophily"]); q = float(p["corruption_fraction"])
    mode = str(p["corruption_mode"])
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(2, dtype=np.int64), n // 2)
    rng.shuffle(labels)
    groups = [np.flatnonzero(labels == c).tolist() for c in range(2)]
    within_slots = max(n // 2 - 1, 1); across_slots = n // 2
    p_in = min(0.98, target_h * target_degree / within_slots)
    p_out = min(0.98, (1.0 - target_h) * target_degree / max(across_slots, 1))
    clean = nx.stochastic_block_model([len(g) for g in groups], [[p_in, p_out], [p_out, p_in]], seed=seed)
    # SBM nodes are block ordered; relabel them to the independently shuffled
    # label locations so node id cannot encode the class.
    block_order = groups[0] + groups[1]
    clean = nx.relabel_nodes(clean, {old: block_order[old] for old in range(n)}, copy=True)
    clean.add_nodes_from(range(n))
    observed = clean.copy()
    if q > 0 and mode == "degree_preserving":
        requested = max(1, int(round(q * observed.number_of_edges())))
        try:
            nx.double_edge_swap(observed, nswap=requested, max_tries=max(100, requested * 100), seed=seed)
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            pass
    elif q > 0 and mode == "homophily_attack":
        _apply_homophily_attack(observed, labels, q, rng)
    elif q > 0:
        raise ValueError(f"Unknown corruption mode: {mode}")
    clean_pairs = {_canon_edge(u, v) for u, v in clean.edges()}
    observed_pairs = {_canon_edge(u, v) for u, v in observed.edges()}
    union = clean_pairs | observed_pairs
    return observed, labels, {
        "controlled_property": "latent_graph_corruption_recovery",
        "labels_generated_before_graph": True,
        "target_latent_homophily": target_h,
        "clean_edge_homophily": _edge_homophily(clean, labels),
        "observed_edge_homophily": _edge_homophily(observed, labels),
        "corruption_fraction_requested": q,
        "corruption_mode": mode,
        "clean_edges": clean.number_of_edges(),
        "observed_edges": observed.number_of_edges(),
        "clean_observed_edge_jaccard": len(clean_pairs & observed_pairs) / max(len(union), 1),
        "actual_changed_edge_fraction": len(clean_pairs ^ observed_pairs) / max(2 * len(clean_pairs), 1),
        "degree_sequence_preserved": sorted(dict(clean.degree()).values()) == sorted(dict(observed.degree()).values()),
    }


def _apply_homophily_attack(graph: nx.Graph, labels: np.ndarray, fraction: float, rng: np.random.Generator) -> None:
    same = [(int(u), int(v)) for u, v in graph.edges() if labels[u] == labels[v]]
    requested = min(len(same), int(round(fraction * graph.number_of_edges())))
    if requested <= 0:
        return
    remove_indices = rng.choice(len(same), size=requested, replace=False)
    graph.remove_edges_from([same[int(i)] for i in np.atleast_1d(remove_indices)])
    left = np.flatnonzero(labels == 0); right = np.flatnonzero(labels == 1)
    added = 0
    while added < requested:
        batch = max(256, 4 * (requested - added))
        us = rng.choice(left, size=batch, replace=True)
        vs = rng.choice(right, size=batch, replace=True)
        for u, v in zip(us, vs):
            if not graph.has_edge(int(u), int(v)):
                graph.add_edge(int(u), int(v)); added += 1
                if added == requested:
                    break


def _canon_edge(u: int, v: int) -> tuple[int, int]:
    return (min(int(u), int(v)), max(int(u), int(v)))


def _edge_homophily(graph: nx.Graph, labels: np.ndarray) -> float:
    if graph.number_of_edges() == 0:
        return float("nan")
    return float(np.mean([labels[int(u)] == labels[int(v)] for u, v in graph.edges()]))


def _balanced_shuffled_labels(n: int, seed: int) -> np.ndarray:
    labels = np.repeat(np.arange(2, dtype=np.int64), n // 2)
    np.random.default_rng(seed).shuffle(labels)
    return labels


def _graph_from_weighted_pairs(n: int, budget: int, weights: np.ndarray, seed: int) -> nx.Graph:
    upper_u, upper_v = np.triu_indices(n, k=1)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Weighted graph construction received invalid edge weights.")
    budget = min(int(budget), len(weights))
    chosen = np.random.default_rng(seed).choice(len(weights), size=budget, replace=False, p=weights / weights.sum())
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(upper_u[i]), int(upper_v[i])) for i in chosen)
    return graph


def _finalize_latent_corruption(
    clean: nx.Graph,
    labels: np.ndarray,
    p: dict[str, object],
    seed: int,
    controlled_property: str,
    extra: dict[str, object],
):
    q = float(p["corruption_fraction"]); mode = str(p["corruption_mode"])
    rng = np.random.default_rng(seed)
    observed = clean.copy()
    if q > 0 and mode == "degree_preserving":
        requested = max(1, int(round(q * observed.number_of_edges())))
        try:
            nx.double_edge_swap(observed, nswap=requested, max_tries=max(100, requested * 100), seed=seed)
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            pass
    elif q > 0 and mode == "homophily_attack":
        _apply_homophily_attack(observed, labels, q, rng)
    elif q > 0:
        raise ValueError(f"Unknown corruption mode: {mode}")
    clean_pairs = {_canon_edge(u, v) for u, v in clean.edges()}
    observed_pairs = {_canon_edge(u, v) for u, v in observed.edges()}
    union = clean_pairs | observed_pairs
    metadata = {
        "controlled_property": controlled_property,
        "labels_generated_before_observed_graph": True,
        "clean_edge_homophily": _edge_homophily(clean, labels),
        "observed_edge_homophily": _edge_homophily(observed, labels),
        "corruption_fraction_requested": q,
        "corruption_mode": mode,
        "clean_edges": clean.number_of_edges(), "observed_edges": observed.number_of_edges(),
        "clean_observed_edge_jaccard": len(clean_pairs & observed_pairs) / max(len(union), 1),
        "actual_changed_edge_fraction": len(clean_pairs ^ observed_pairs) / max(2 * len(clean_pairs), 1),
        "degree_sequence_preserved": sorted(dict(clean.degree()).values()) == sorted(dict(observed.degree()).values()),
        **extra,
    }
    return observed, labels, metadata


def _latent_geometric_recovery(p: dict[str, object], seed: int):
    n = int(p["n"]); degree = float(p["target_avg_degree"]); separation = float(p["geometric_separation"])
    labels = _balanced_shuffled_labels(n, seed)
    rng = np.random.default_rng(seed)
    centers = np.column_stack(((2 * labels - 1) * separation / 2.0, np.zeros(n)))
    positions = centers + rng.normal(size=(n, 2))
    upper_u, upper_v = np.triu_indices(n, k=1)
    distances = np.linalg.norm(positions[upper_u] - positions[upper_v], axis=1)
    budget = max(n - 1, int(round(n * degree / 2)))
    chosen = np.argpartition(distances, min(budget, len(distances)) - 1)[:budget]
    clean = nx.Graph(); clean.add_nodes_from(range(n))
    clean.add_edges_from((int(upper_u[i]), int(upper_v[i])) for i in chosen)
    return _finalize_latent_corruption(
        clean, labels, p, seed, "unseen_latent_geometric_recovery",
        {"geometric_separation": separation, "position_dimensions": 2},
    )


def _latent_degree_corrected_recovery(p: dict[str, object], seed: int):
    n = int(p["n"]); degree = float(p["target_avg_degree"])
    homophily = float(p["latent_homophily"]); exponent = float(p["degree_exponent"])
    labels = _balanced_shuffled_labels(n, seed)
    rng = np.random.default_rng(seed)
    theta = np.minimum(rng.pareto(max(exponent - 1.0, 0.2), size=n) + 1.0, 12.0)
    upper_u, upper_v = np.triu_indices(n, k=1)
    same = labels[upper_u] == labels[upper_v]
    block_weight = np.where(same, max(homophily, 0.02), max(1.0 - homophily, 0.02))
    weights = theta[upper_u] * theta[upper_v] * block_weight
    clean = _graph_from_weighted_pairs(n, int(round(n * degree / 2)), weights, seed)
    return _finalize_latent_corruption(
        clean, labels, p, seed, "unseen_degree_corrected_recovery",
        {"target_latent_homophily": homophily, "degree_exponent": exponent,
         "clean_degree_cv": float(np.std([d for _, d in clean.degree()]) / max(np.mean([d for _, d in clean.degree()]), 1e-12))},
    )


def _latent_hierarchical_recovery(p: dict[str, object], seed: int):
    n = int(p["n"]); degree = float(p["target_avg_degree"])
    homophily = float(p["latent_homophily"]); strength = float(p["hierarchy_strength"])
    rng = np.random.default_rng(seed)
    communities = np.repeat(np.arange(4, dtype=np.int64), n // 4)
    rng.shuffle(communities)
    labels = (communities >= 2).astype(np.int64)
    upper_u, upper_v = np.triu_indices(n, k=1)
    same_subcommunity = communities[upper_u] == communities[upper_v]
    same_label = labels[upper_u] == labels[upper_v]
    cross_weight = max(1.0 - homophily, 0.02)
    weights = np.where(same_subcommunity, strength * max(homophily, 0.02),
                       np.where(same_label, max(homophily, 0.02), cross_weight))
    clean = _graph_from_weighted_pairs(n, int(round(n * degree / 2)), weights, seed)
    return _finalize_latent_corruption(
        clean, labels, p, seed, "unseen_hierarchical_recovery",
        {"target_latent_homophily": homophily, "hierarchy_strength": strength, "subcommunities": 4},
    )


def _degree_matched_bottleneck(p: dict[str, object], seed: int):
    n = int(p["n"]); d = int(p["degree"]); requested = int(p["bridges"])
    half = n // 2
    left = nx.random_regular_graph(d, half, seed=seed)
    right = nx.relabel_nodes(nx.random_regular_graph(d, half, seed=stable_seed(seed, "right")), lambda x: x + half)
    graph = nx.compose(left, right)
    rng = np.random.default_rng(seed)
    swaps = max(1, math_ceil_div(requested, 2))
    completed = 0
    for _ in range(swaps * 30):
        if completed >= swaps:
            break
        e1 = list(left.edges())[int(rng.integers(0, left.number_of_edges()))]
        e2 = list(right.edges())[int(rng.integers(0, right.number_of_edges()))]
        a, b = map(int, e1); c, d2 = map(int, e2)
        new_edges = ((a, c), (b, d2)) if rng.random() < 0.5 else ((a, d2), (b, c))
        if any(graph.has_edge(*edge) for edge in new_edges):
            continue
        graph.remove_edge(a, b); graph.remove_edge(c, d2)
        graph.add_edges_from(new_edges)
        left.remove_edge(a, b); right.remove_edge(c, d2)
        completed += 1
    labels = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64)])
    return graph, labels, {
        "controlled_property": "bottleneck_at_exact_degree_sequence",
        "requested_bridge_edges": requested,
        "actual_bridge_edges": int(sum((u < half) != (v < half) for u, v in graph.edges())),
        "degree_sequence_preserved": True,
    }


def math_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _spectral_labels(graph: nx.Graph, classes: int, seed: int) -> np.ndarray:
    n = graph.number_of_nodes()
    if n < classes:
        return np.arange(n, dtype=np.int64)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=float, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    safe = degree.copy(); safe[safe == 0] = 1.0
    lap = sparse.eye(n, format="csr") - sparse.diags(1.0 / np.sqrt(safe)) @ adjacency @ sparse.diags(1.0 / np.sqrt(safe))
    try:
        _, vectors = eigsh(lap, k=min(max(classes, 2), n - 1), which="SM", tol=1e-4)
        embedding = vectors[:, 1:min(classes, vectors.shape[1])]
        if embedding.ndim == 1:
            embedding = embedding[:, None]
        return KMeans(n_clusters=classes, n_init=10, random_state=seed).fit_predict(embedding).astype(np.int64)
    except Exception:
        nodes = np.arange(n)
        return (nodes * classes // max(n, 1)).clip(max=classes - 1).astype(np.int64)


def _ensure_viable_labels(graph: nx.Graph, labels: np.ndarray, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    if len(counts) >= 2 and counts.min() >= 2:
        return labels
    return _spectral_labels(graph, 2, seed)


def generate_tasks(graph: SyntheticGraph, cfg: TaskDesignConfig, master_seed: int) -> Iterable[SyntheticTask]:
    labels = graph.structural_labels.astype(np.int64, copy=True)
    if np.bincount(labels).min() < cfg.min_class_size:
        labels = _spectral_labels(graph.graph, 2, stable_seed(master_seed, graph.graph_id, "fallback_labels"))
    classes = int(labels.max()) + 1
    for regime in cfg.feature_regimes:
        signal = float(cfg.class_signal[regime])
        for task_index in range(cfg.task_seeds_per_graph):
            task_seed = stable_seed(master_seed, graph.graph_id, regime, task_index)
            rng = np.random.default_rng(task_seed)
            prototypes = rng.normal(size=(classes, cfg.feature_dimensions))
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            prototypes = prototypes / np.maximum(norms, 1e-12)
            noise = rng.normal(scale=cfg.feature_noise_std, size=(len(labels), cfg.feature_dimensions))
            features = signal * prototypes[labels] + noise
            features = features.astype(np.float32)
            task_id = f"{graph.graph_id}__{regime}__t{task_index:02d}"
            yield SyntheticTask(task_id, graph.graph_id, regime, task_seed, features, labels.copy())
