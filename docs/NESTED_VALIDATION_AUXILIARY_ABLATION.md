# Auxiliary GCN and disjoint-validation ablation

This experiment answers two separate threats to validity in the original
protocol.

1. It evaluates the supervised auxiliary GCN directly as a node classifier.
2. It prevents the observations used for checkpoint selection from also being
   used to choose the reconstructed topology.

## Split and decision order

Each seed creates one stratified, exhaustive and disjoint split:

- 60% train;
- 10% inner validation;
- 10% topology validation;
- 20% test.

The auxiliary GCN is trained on `train` and checkpointed on `inner-validation`.
Its restored checkpoint supplies both its classifier-only score and the hidden
representation used by UMAP and Delaunay. Downstream GNN checkpoints are also
selected on `inner-validation`. The reconstructed graph and rewiring ratio are
selected exclusively by downstream macro-F1 on `topology-validation`. Test is
read only after both decisions have been fixed.

Existing tuned downstream hyperparameters and unsupervised VGAE embeddings are
treated as fixed inputs. They are not retuned in this targeted ablation.

## Targeted pilot

The default pilot uses four deliberately heterogeneous datasets:

- Airports-USA: strong positive effect in the original real-data study;
- Cora: approximately neutral effect;
- Amazon-Photo: negative effect;
- Roman-Empire: heterophilous graph with strong method/backbone interaction.

It runs GCN, GraphSAGE and GAT for 10 or more seeds. This pilot tests whether the
main conclusion survives the stricter decision boundary; it is not presented as
a replacement for a full 12-dataset rerun.

## Outputs

- `nested_auxiliary_selected_per_seed.csv`: one row per dataset/backbone/seed,
  including Original GNN, Auxiliary GCN only and validation-selected reconstructed
  graph results.
- `nested_auxiliary_candidates_long.csv`: every reconstructed candidate before
  topology selection.
- `nested_auxiliary_summary.csv`: paired gains, bootstrap confidence intervals,
  positive-run rates and paired Wilcoxon tests.

The runner is resumable and supports independent workers through `--num-shards`
and `--shard-index`.
