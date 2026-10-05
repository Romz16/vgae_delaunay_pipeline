# Synthetic Structural Rewiring Suitability Analysis

Graphs analyzed: 205
Families: 6
Configurations: 41

## Frozen indicator

Model: two_metric_logistic__lambda2_norm_laplacian__density
Features: lambda2_norm_laplacian, density
Recommend threshold: 0.0592
Avoid threshold: 0.9900

## Highest exploratory importances

- lambda2_norm_laplacian: importance=0.0000, direction=-0.3530, robustness=0.998
- algebraic_connectivity: importance=0.0000, direction=-0.2835, robustness=1.000
- spectral_sweep_conductance: importance=0.0000, direction=-0.3341, robustness=0.996
- edge_connectivity_ratio: importance=0.0000, direction=0.4280, robustness=0.996
- edge_connectivity: importance=0.0000, direction=0.3229, robustness=0.988
- effective_resistance_sample: importance=0.0000, direction=0.1826, robustness=0.884
- num_isolates: importance=0.0000, direction=-0.0126, robustness=0.850
- largest_cc_ratio: importance=0.0000, direction=0.0126, robustness=0.824

## Interpretation rule

Post-rewiring metrics explain mechanisms but are excluded from the indicator. Real-dataset outcomes are external validation only and must never be used to modify the frozen model or thresholds.
