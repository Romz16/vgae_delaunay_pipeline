
from __future__ import annotations

import unittest

try:
    import networkx as nx
    import numpy as np
    from src.synthetic_suitability.config import profile_config
    from src.synthetic_suitability.generation import GraphSpecification, generate_one
    from src.synthetic_suitability.structural import INDICATOR_FORBIDDEN_COLUMNS, compute_prerewiring_metrics
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "scientific Python dependencies are not installed")
class SyntheticDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = profile_config("smoke")

    def test_degree_matched_bottleneck_preserves_degrees(self):
        spec = GraphSpecification(
            family="degree_matched_bottleneck",
            configuration_id="test",
            parameters={"n": 90, "degree": 6, "bridges": 12},
        )
        graph, labels, metadata = generate_one(spec, 123, self.config.graph)
        self.assertEqual(set(dict(graph.degree()).values()), {6})
        self.assertEqual(len(labels), 90)
        self.assertTrue(metadata["degree_sequence_preserved"])
        self.assertGreaterEqual(metadata["actual_bridge_edges"], 12)

    def test_prerewiring_features_exclude_leakage(self):
        graph = nx.watts_strogatz_graph(90, 6, 0.2, seed=5)
        metrics = compute_prerewiring_metrics(graph, self.config.metrics, seed=5)
        self.assertFalse(INDICATOR_FORBIDDEN_COLUMNS.intersection(metrics))
        self.assertIn("lambda2_norm_laplacian", metrics)
        self.assertIn("spectral_sweep_conductance", metrics)

    def test_density_matched_sbm_keeps_expected_degree(self):
        observed=[]
        for mixing in (0.03, 0.45):
            spec=GraphSpecification("sbm_density_matched",f"m{mixing}",{"n":240,"target_avg_degree":12,"communities":4,"mixing":mixing})
            graph,_,_=generate_one(spec,77,self.config.graph)
            observed.append(2*graph.number_of_edges()/graph.number_of_nodes())
        self.assertLess(abs(observed[0]-observed[1]),3.5)


if __name__ == "__main__":
    unittest.main()
