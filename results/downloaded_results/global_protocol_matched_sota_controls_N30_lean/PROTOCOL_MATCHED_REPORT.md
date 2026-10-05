# Protocol-matched SOTA and constructor controls

Status: **PASS**

- Same 12 dataset loaders/versions and stratified 60/20/20 splits.
- 30 common seeds starting at 12345; checkpoint and candidate selection by validation macro-F1.
- Same saved 30-trial GNN tuning results and 400 final epochs as the proposed method.
- SDRF official sparse Balanced-Forman algorithm; candidate step budgets selected on validation.
- DiffWire CT evaluated on the official GCN comparison backbone using scalable spectral CTE edge relevance.
- kNN, mutual-kNN, random edge-budget, and random degree-matched run on all 12 datasets using identical auxiliary-GCN UMAP coordinates and the exact Delaunay edge budget.
- Radius and MST+kNN are supplementary controls on representative datasets: Actor, Cora, Pubmed, Roman-Empire.
