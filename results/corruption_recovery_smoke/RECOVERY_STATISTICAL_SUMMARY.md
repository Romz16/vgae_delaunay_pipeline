# Corruption–recovery: primary statistical analysis

The independent graph instance is the inferential unit. Feature/pipeline repetitions are averaged 
before the paired Wilcoxon test; confidence intervals use a graph-level bootstrap. Holm correction 
controls multiplicity across constructors and ratios.

| Constructor | Ratio | Mean gain (pp) | 95% CI | Positive graphs | Holm p |
|---|---:|---:|---:|---:|---:|
| raw_umap_delaunay | 0.00 | 34.286 | [34.286, 34.286] | 100.0% | 1 |
| random_edge_budget_matched | 0.00 | 26.805 | [26.805, 26.805] | 100.0% | 1 |
| knn | 0.00 | 22.857 | [22.857, 22.857] | 100.0% | 1 |
| random_degree_matched | 0.00 | 22.857 | [22.857, 22.857] | 100.0% | 1 |
| mst_knn | 0.00 | 22.857 | [22.857, 22.857] | 100.0% | 1 |
| delaunay | 0.25 | 19.494 | [19.494, 19.494] | 100.0% | 1 |
| learned_hd_knn | 0.00 | 19.048 | [19.048, 19.048] | 100.0% | 1 |
| mutual_knn | 0.00 | 18.214 | [18.214, 18.214] | 100.0% | 1 |
| delaunay_union_original | 0.00 | 18.214 | [18.214, 18.214] | 100.0% | 1 |
| delaunay | 0.00 | 15.584 | [15.584, 15.584] | 100.0% | 1 |
| raw_hd_knn | 0.00 | 11.429 | [11.429, 11.429] | 100.0% | 1 |
