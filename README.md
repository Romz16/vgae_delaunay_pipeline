# Embedding-Guided Delaunay Rewiring

This repository contains the reproducible implementation and curated experimental
artifacts for embedding-guided graph rewiring in node-classification tasks.

The proposed pipeline learns a task-informed node representation, projects it
with UMAP, builds a geometric Delaunay graph, optionally refines it with a VGAE,
and selects the topology using validation data only before final test evaluation.

## Repository layout

```text
src/        Core pipeline, graph constructors, models, and evaluation utilities
scripts/    Reproducible experiment and analysis entry points
tests/      Automated tests and smoke-test helpers
docs/       Methodological notes and experiment documentation
results/    Curated per-seed outputs, analyses, and result inventories
```

## Method overview

1. Train an auxiliary GCN to learn a task-informed representation from the
   training labels.
2. Use UMAP to obtain a two-dimensional geometric layout of the learned
   embeddings.
3. Construct a Delaunay graph from that layout.
4. Optionally combine the geometric graph with VGAE-based edge refinement.
5. Select a candidate topology using validation F1 only; the test split remains
   unseen until the final evaluation.
6. Train and evaluate the downstream GNN on the selected topology.

The `Delaunay + VGAE 0%` condition is the pure geometric Delaunay graph. It is
not the original-graph baseline, which is always reported separately.

## Installation

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run a dataset experiment

```bash
python main.py --dataset Cora --root ./data --output-dir outputs/cora
```

Examples of supported PyG datasets include `Cora`, `Citeseer`, `Pubmed`,
`Cornell`, `Texas`, `Wisconsin`, `Chameleon`, `Squirrel`, `Actor`, `WikiCS`,
`Flickr`, `LastFMAsia`, and the Airports datasets.

The pipeline can also load a graph from `.pt`, `.pth`, `.npz`, GraphML, GEXF,
GPickle, or a CSV edge list. For node classification, labels are required; if
node features are absent, degree-based features are generated.

## Fast smoke test

```bash
python main.py \
  --dataset Wisconsin \
  --output-dir outputs/wisconsin_smoke \
  --vgae-trials 3 \
  --gnn-trials 3 \
  --vgae-opt-epochs 20 \
  --vgae-final-epochs 20 \
  --gnn-opt-epochs 20 \
  --gnn-final-epochs 20 \
  --final-runs 2
```

## Published results

The [`results/`](results/) directory contains the per-seed experimental data and
derived analyses used in the study, including global 12-dataset runs, robustness
analyses, synthetic and corruption experiments, matched baseline controls, and
the Citeseer/Wisconsin extension. See [`results/README.md`](results/README.md)
for an inventory and interpretation caveats.

## Reproducibility principles

- Dataset versions, splits, seeds, tuning budgets, and validation selection are
  recorded in the experiment artifacts.
- Candidate selection is validation-only; no candidate is selected by test F1.
- Reported SDRF and DiffWire-CT controls are protocol-matched adaptations, not
  unchanged executions of their upstream repositories.
