from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path("outputs_revision/downloaded_results/corruption_recovery_pilot_20260918")
g = pd.read_csv(root / "graph_level_outcomes.csv")
metadata = g.generation_metadata.map(json.loads)
g["observed_homophily"] = metadata.map(lambda x: x["observed_edge_homophily"])
g["gain_pp"] = 100 * g.gain_selected_rewiring
rng = np.random.default_rng(20260918)

metrics = [
    "observed_homophily", "lambda2_norm_laplacian", "density", "degree_cv",
    "avg_clustering", "effective_resistance_sample", "modularity",
]
rows = []
for metric in metrics:
    quartiles = pd.qcut(g[metric], 4, duplicates="drop")
    for level, part in g.groupby(quartiles, observed=True):
        values = part.gain_pp.to_numpy()
        bootstrap = np.array([
            rng.choice(values, len(values), replace=True).mean() for _ in range(5000)
        ])
        rows.append({
            "metric": metric, "quartile": str(level), "n": len(values),
            "mean_gain_pp": values.mean(), "median_gain_pp": np.median(values),
            "ci95_low_pp": np.quantile(bootstrap, 0.025),
            "ci95_high_pp": np.quantile(bootstrap, 0.975),
            "positive_fraction": np.mean(values > 0),
        })
quartile_results = pd.DataFrame(rows).sort_values("mean_gain_pp", ascending=False)
quartile_results.to_csv(root / "analysis_structural_metric_quartiles.csv", index=False)

combination = g.groupby(
    ["param_n", "param_target_avg_degree", "param_latent_homophily", "param_corruption_fraction"],
    as_index=False,
).agg(n=("gain_pp", "size"), mean_gain_pp=("gain_pp", "mean"), median_gain_pp=("gain_pp", "median"))
combination.to_csv(root / "analysis_generation_factor_combinations.csv", index=False)

print("TOP QUARTILES")
print(quartile_results.head(15).to_string(index=False))
print("\nTOP GENERATION COMBINATIONS")
print(combination.sort_values("mean_gain_pp", ascending=False).head(15).to_string(index=False))
