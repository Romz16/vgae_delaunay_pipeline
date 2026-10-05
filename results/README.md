# Results inventory

This directory contains the curated artifacts behind the reported experiments.
Raw per-seed CSV files are retained whenever available so aggregate statistics
can be independently recomputed.

## Main experiment families

| Directory | Contents |
| --- | --- |
| `downloaded_results/` | Global 12-dataset experimental table and recovered source results. |
| `robustness_gate_analysis_20261002/` | Hierarchical bootstrap, leave-one-dataset-out, selection-regret, and matched fallback analyses. |
| `synthetic_*` | Controlled synthetic-graph experiments. |
| `corruption_*` | Original-graph perturbation and recovery experiments. |
| `extension_citeseer_wisconsin_20261005/` | Independent 100-seed, three-backbone extensions for Citeseer and Wisconsin. |
| `idgl_*` | IDGL compatibility pilots. These are exploratory and are not presented as an unchanged official IDGL baseline. |

## Reading the data

The preferred long-format files contain one row per dataset, backbone, seed, and
candidate topology. They expose validation and test F1 separately. The selected
candidate is chosen from validation performance; test F1 is consulted only after
selection.

For each seed, the minimum fields needed to recompute the principal comparison
are:

```text
dataset, backbone, seed,
original_test_f1, selected_test_f1, selected_candidate
```

## Important interpretation notes

- The global and extension analyses compare the selected proposed topology with
  the original graph baseline using the same split and seed.
- Pure Delaunay corresponds to the zero VGAE-refinement rate. Positive rates
  add the second-stage VGAE refinement.
- The SDRF and DiffWire-CT controls are scalable, protocol-matched adaptations
  based on those methods. They are not unchanged runs of the upstream codebases.
- The IDGL artifacts are compatibility pilots, not a claim of a fully native
  official-IDGL reproduction.

Historical artifact names are kept unchanged to preserve provenance and ensure
that scripts referring to the files remain reproducible.
