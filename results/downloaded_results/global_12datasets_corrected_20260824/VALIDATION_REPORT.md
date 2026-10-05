# Delivery Validation Report

Overall status: **PASS**

- [PASS] selected_rates.csv: rows=3600; datasets=12; dataset_column=True
- [PASS] all_runs_long.csv: rows=21600; datasets=12; dataset_column=True
- [PASS] final_selected_test_results.csv: rows=36; datasets=12; dataset_column=True
- [PASS] raw_feature_delaunay_results.csv: rows=36; datasets=12; dataset_column=True
- [PASS] raw_feature_delaunay_all_runs.csv: rows=3600; datasets=12; dataset_column=True
- [PASS] raw_feature_delaunay_comparison.csv: rows=3600; datasets=12; dataset_column=True
- [PASS] umap_trustworthiness.csv: rows=24; datasets=12; dataset_column=True
- [PASS] selected_before_after_structural_metrics.csv: rows=3600; datasets=12; dataset_column=True
- [PASS] original_graph_metrics.csv: rows=12; datasets=12; dataset_column=True
- [PASS] selected_rewired_graph_metrics.csv: rows=84; datasets=12; dataset_column=True
- [PASS] structural_metric_correlations.csv: rows=24
- [PASS] structural_correlation_sensitivity.csv: rows=112
- [PASS] density_confounder_analysis.csv: rows=7
- [PASS] baseline_saturation_analysis.csv: rows=1
- [PASS] structural_delta_correlations.csv: rows=14
- [PASS] paired_statistical_tests.csv: rows=36
- [PASS] UMAP trustworthiness completeness: missing_trustworthiness=0; sources=2; two_sources_per_dataset=True
- [PASS] global figures: png_files=9

## Scientific caveat

Missing effective-resistance values: 3.
These absences reflect numerical solver nonconvergence. The primary lambda2, Cheeger, density and homophily analyses are complete for all 12 datasets.

No model retraining was performed for this correction; only the aggregation metadata and delivery validation were regenerated.
