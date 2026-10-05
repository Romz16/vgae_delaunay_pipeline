"""End-to-end smoke test for leakage-safe indicator fitting."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from src.synthetic_suitability.config import profile_config
    from src.synthetic_suitability.indicator import fit_indicator
    from src.synthetic_suitability.reporting import generate_outputs
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "scientific Python dependencies are not installed")
class IndicatorSmokeTest(unittest.TestCase):
    def test_indicator_freezes_without_real_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            base = profile_config("smoke", output)
            indicator = replace(
                base.indicator,
                candidate_features=("lambda2_norm_laplacian", "density", "approx_avg_shortest_path"),
                external_compatible_features=("lambda2_norm_laplacian", "density", "approx_avg_shortest_path"),
                bootstrap_repetitions=5,
                min_leaf_graphs=3,
            )
            config = replace(base, indicator=indicator)
            rng = np.random.default_rng(8)
            rows=[]
            for index in range(80):
                lambda2=float(rng.uniform(0,0.25)); path=float(rng.uniform(2,9)); density=float(rng.uniform(.005,.08))
                gain=0.10*lambda2-0.003*(path-4)+0.02*density+rng.normal(0,.004)
                rows.append({
                    "graph_id":f"g{index}", "family":f"f{index%4}", "configuration_id":f"c{index%16}",
                    "lambda2_norm_laplacian":lambda2, "density":density, "approx_avg_shortest_path":path,
                    "spectral_sweep_conductance":float(np.clip(lambda2+rng.normal(0,.02),0,1)),
                    "gain_selected_rewiring":gain, "gain_geometry":gain*.7, "gain_vgae":gain*.3,
                    "success_0_5":int(100*gain>.5),
                })
            pd.DataFrame(rows).to_csv(output/"graph_level_outcomes.csv",index=False)
            manifest=fit_indicator(config)
            generate_outputs(config)
            self.assertFalse(manifest["external_real_data_used_for_fit"])
            self.assertTrue((output/"frozen_suitability_indicator.joblib").exists())
            self.assertTrue((output/"gain_regression_metrics.csv").exists())
            self.assertTrue((output/"figures"/"01_lambda2_vs_rewiring_gain.png").exists())
            self.assertTrue((output/"manuscript"/"Structural_Rewiring_Suitability.md").exists())
            self.assertGreater(manifest["fit_graphs"],50)


if __name__ == "__main__":
    unittest.main()
