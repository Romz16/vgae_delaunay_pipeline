# Corruption–recovery: primary statistical analysis

The independent graph instance is the inferential unit. Feature/pipeline repetitions are averaged 
before the paired Wilcoxon test; confidence intervals use a graph-level bootstrap. Holm correction 
controls multiplicity across constructors and ratios.

| Constructor | Ratio | Mean gain (pp) | 95% CI | Positive graphs | Holm p |
|---|---:|---:|---:|---:|---:|
| raw_hd_knn | 0.00 | 6.910 | [6.353, 7.512] | 93.9% | 5.259e-29 |
| raw_umap_delaunay | 0.00 | 4.420 | [3.836, 5.027] | 86.7% | 2.791e-23 |
| learned_hd_knn | 0.00 | 0.550 | [0.027, 1.070] | 53.9% | 0.1469 |
| random_degree_matched | 0.00 | 0.252 | [-0.337, 0.821] | 50.6% | 1 |
| random_edge_budget_matched | 0.00 | -0.127 | [-0.737, 0.468] | 47.2% | 1 |
| delaunay | 0.55 | -0.269 | [-0.776, 0.200] | 49.4% | 1 |
| delaunay | 0.40 | -0.439 | [-0.917, 0.023] | 51.1% | 0.6093 |
| delaunay | 0.25 | -1.012 | [-1.495, -0.531] | 37.2% | 0.0006613 |
| delaunay_union_original | 0.00 | -1.091 | [-1.509, -0.681] | 31.7% | 3.543e-06 |
| delaunay | 0.10 | -1.516 | [-2.051, -0.985] | 31.7% | 8.316e-07 |
| delaunay | 0.00 | -2.552 | [-3.037, -2.021] | 25.6% | 1.466e-15 |
| knn | 0.00 | -2.887 | [-3.411, -2.368] | 19.4% | 4.566e-18 |
| mst_knn | 0.00 | -2.929 | [-3.443, -2.425] | 18.9% | 4.032e-18 |
| mutual_knn | 0.00 | -3.082 | [-3.597, -2.589] | 16.7% | 4.487e-21 |
