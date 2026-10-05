# Synthetic Structural Rewiring Suitability Experiment

## Scientific questions

The experiment answers two preregistered questions.

1. Can properties of the original unlabeled topology predict whether task-informed geometric rewiring improves downstream node classification?
2. Can the same properties predict whether VGAE similarity refinement adds value beyond learned geometric reconstruction?

The indicator never receives baseline F1, node labels, post-rewiring metrics, validation outcomes, or test gains as inputs. Test gain is a target only. The real datasets are excluded from model selection, coefficient fitting, feature selection, calibration, and threshold selection.

## Experimental units

The hierarchy is:

```text
graph family
  structural configuration
    independent graph instance
      feature and label realization
        pipeline seed
```

The independent graph instance is the unit used to fit the indicator. Feature realizations and pipeline seeds are aggregated within each graph. Grouped cross-validation holds out complete structural configurations. A leave-one-family-out analysis measures transfer to unseen graph generators.

## Controlled graph families

- Density-matched SBM varies community mixing while holding node count and expected degree fixed.
- Watts-Strogatz varies shortcut probability while holding node count and degree fixed.
- Degree-matched bottleneck graphs vary the number of bridges through degree-preserving edge swaps.
- Barabási-Albert varies degree heterogeneity.
- Erdős-Rényi provides a simple random-connectivity baseline.
- Random regular graphs provide a homogeneous-degree control.

Every configuration stores its generating parameters in separate columns. This supports matched comparisons within a family instead of relying only on pooled correlations.

## Node-classification tasks

Labels follow planted communities when the generator supplies them. Other families use a deterministic spectral partition of the original graph. Node attributes combine class prototypes with Gaussian noise under informative, partially informative, and noisy regimes. Labels remain fixed within a graph instance, while feature realizations vary independently.

The indicator is topology-only even though the treatment is task-informed. This deliberate restriction tests how much can be known before training the rewiring pipeline and creates an honest uncertainty region when topology cannot determine feature quality.

## Rewiring protocol

Each nested run uses the same split and downstream training seed for all graph conditions.

1. Train the auxiliary GCN using train labels only.
2. Project its learned representation to two dimensions with UMAP.
3. Construct learned Delaunay, kNN, mutual-kNN, random edge-budget-matched, and random degree-matched graphs.
4. Train VGAE on the original graph.
5. Evaluate Delaunay refinement rates 0, 0.10, 0.25, 0.40, and 0.55.
6. Select the rate using validation macro-F1 only.
7. Read the selected test score after selection.

The primary outcome is the validation-selected Delaunay plus VGAE result against the original graph. The mechanism decomposition is:

```text
gain_geometry = F1 learned Delaunay at r=0 minus F1 original
gain_VGAE = F1 validation-selected positive refinement minus F1 learned Delaunay at r=0
```

## Resource profiles

The CLI provides three fixed profiles.

- `smoke` validates software only and is not scientifically interpretable.
- `pilot` verifies effect directions and runtime before committing full resources.
- `full` uses at least 10 independent instances per structural configuration and the paper training epochs.

Run `init` first. It writes `design_summary.json` with exact graph counts, nested runs, and estimated downstream fits.

## Commands

```powershell
python scripts/run_synthetic_suitability.py init --profile pilot
python scripts/run_synthetic_suitability.py catalog --profile pilot
python scripts/run_synthetic_suitability.py run --profile pilot
python scripts/run_synthetic_suitability.py fit --profile pilot
python scripts/run_synthetic_suitability.py validate-real --profile pilot
python scripts/run_synthetic_suitability.py report --profile pilot
```

Use the same `--output-dir` or `--config` for every phase. Execution writes one atomic part file per nested run, so interrupted jobs can resume safely. `--overwrite` discards resume behavior only for the requested phase; it does not change the preregistered design.

## Output contract

The principal files are:

- `synthetic_graph_metrics.csv` for original-topology features;
- `all_execution_results.csv` for every condition;
- `selected_execution_results.csv` for validation-selected outcomes;
- `graph_level_outcomes.csv` for the independent analysis units;
- `configuration_level_outcomes.csv` for controlled structural summaries;
- `postrewiring_deltas.csv` for mechanism analysis only;
- `frozen_suitability_indicator.json` and `.joblib`;
- `frozen_vgae_refinement_indicator.json` and `.joblib`;
- `indicator_model_comparison.csv` and grouped out-of-fold predictions;
- `external_real_dataset_validation.csv` produced only after freezing;
- article figures, tables, and manuscript subsections.

## Statistical safeguards

- Pipeline seeds are never counted as independent graphs.
- Cross-validation groups complete structural configurations.
- False-positive recommendations receive three times the cost of false negatives by default.
- The final decision has Recommend, Uncertain, and Avoid regions derived from synthetic out-of-fold probabilities.
- A random forest is exploratory and provides permutation importance; the frozen indicator is selected from interpretable models.
- Cluster bootstrap samples structural configurations, not individual runs.
- External real-data evaluation cannot update coefficients or thresholds.

## Interpretation

The study may conclude that no stable topology-only indicator exists. That is a valid result because task-informed rewiring also depends on feature-label alignment, which is intentionally unavailable to the indicator. Generalization failure from synthetic to real graphs must be reported without refitting.
