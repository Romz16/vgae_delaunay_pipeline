"""Dataset and graph-file loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    Airports,
    Amazon,
    Coauthor,
    Flickr,
    LastFMAsia,
    Planetoid,
    WebKB,
    WikiCS,
    WikipediaNetwork,
)
from torch_geometric.transforms import NormalizeFeatures

try:
    from torch_geometric.datasets import HeterophilousGraphDataset
except ImportError:  # pragma: no cover
    HeterophilousGraphDataset = None

from src.data.graph_utils import networkx_to_pyg, normalize_pyg_data


class GraphLoader:
    """Load graphs from PyG dataset names or local files.

    The pipeline expects a PyG ``Data`` object with ``edge_index``, ``x`` and
    ``y``. For files without features, simple degree-based features are created
    later by the graph normalization step.
    """

    def __init__(self, root: str | Path = "./data", device: torch.device | None = None) -> None:
        """Initialize the loader.

        Args:
            root: Root directory used by PyG datasets.
            device: Target PyTorch device.
        """
        self.root = Path(root)
        self.device = device or torch.device("cpu")

    def load_from_name(self, name: str) -> Data:
        """Load a supported PyG dataset by name.

        Args:
            name: Dataset identifier, such as ``Cora``, ``Wisconsin``,
                ``LastFMAsia``, ``Roman-empire`` or ``Airports-Brazil``.

        Returns:
            Normalized PyG graph.
        """
        key = name.lower().replace("_", "-")
        transform = NormalizeFeatures()

        if key in {"cora", "pubmed", "citeseer"}:
            canonical_name = {"cora": "Cora", "pubmed": "PubMed", "citeseer": "CiteSeer"}[key]
            dataset = Planetoid(root=str(self.root / "Planetoid"), name=canonical_name, transform=transform)
        elif key in {"cornell", "texas", "wisconsin"}:
            dataset = WebKB(root=str(self.root / "WebKB"), name=name.capitalize(), transform=transform)
        elif key in {"actor"}:
            dataset = Actor(root=str(self.root / "Actor"), transform=transform)
        elif key in {"wikics", "wiki-cs"}:
            dataset = WikiCS(root=str(self.root / "WikiCS"), transform=transform)
        elif key in {"lastfmasia", "lastfm-asia"}:
            dataset = LastFMAsia(root=str(self.root / "LastFMAsia"), transform=transform)
        elif key in {"flickr"}:
            dataset = Flickr(root=str(self.root / "Flickr"), transform=transform)
        elif key in {"chameleon", "squirrel", "crocodile"}:
            dataset = WikipediaNetwork(
                root=str(self.root / "WikipediaNetwork"),
                name=key,
                transform=transform,
            )
        elif key in {"airports-brazil", "airport-br", "brazil", "airports-br"}:
            dataset = Airports(root=str(self.root / "Airports"), name="Brazil", transform=transform)
        elif key in {"airports-usa", "usa", "airports-us"}:
            dataset = Airports(root=str(self.root / "Airports"), name="USA", transform=transform)
        elif key in {"airports-europe", "europe"}:
            dataset = Airports(root=str(self.root / "Airports"), name="Europe", transform=transform)
        elif key in {"amazon-computers", "amazon-photo"}:
            amazon_name = "Computers" if "computers" in key else "Photo"
            dataset = Amazon(root=str(self.root / "Amazon"), name=amazon_name, transform=transform)
        elif key in {"coauthor-cs", "coauthorcs", "coauthor"}:
            dataset = Coauthor(root=str(self.root / "coauthor"), name="CS", transform=transform)
        elif key in {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}:
            if HeterophilousGraphDataset is None:
                raise ImportError("Sua versão do PyG não possui HeterophilousGraphDataset.")
            dataset = HeterophilousGraphDataset(
                root=str(self.root / "HeterophilousGraphDataset"),
                name=key,
                transform=transform,
            )
        else:
            raise ValueError(f"Dataset PyG não suportado: {name}")

        return normalize_pyg_data(dataset[0], self.device)

    def load_from_file(self, path: str | Path, file_format: str | None = None) -> Data:
        """Load a graph from a local file.

        Supported formats:
            - ``.pt``/``.pth``: saved PyG ``Data`` or dict with tensors.
            - ``.npz``: arrays ``edge_index``, ``x`` and ``y``.
            - ``.csv``: edge list with columns source,target; optional label file
              is not inferred automatically.
            - ``.graphml``, ``.gexf``, ``.gpickle``: NetworkX graphs.

        Args:
            path: Input graph path.
            file_format: Optional explicit format string.

        Returns:
            Normalized PyG graph.
        """
        path = Path(path)
        fmt = (file_format or path.suffix.lstrip(".")).lower()
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de grafo não encontrado: {path}")

        if fmt in {"pt", "pth"}:
            obj = torch.load(path, map_location="cpu")
            data = self._from_torch_object(obj)
        elif fmt == "npz":
            data = self._from_npz(path)
        elif fmt == "csv":
            data = self._from_edge_csv(path)
        elif fmt == "graphml":
            data = networkx_to_pyg(nx.read_graphml(path))
        elif fmt == "gexf":
            data = networkx_to_pyg(nx.read_gexf(path))
        elif fmt in {"gpickle", "pickle", "pkl"}:
            data = networkx_to_pyg(nx.read_gpickle(path))
        else:
            raise ValueError(f"Formato de grafo não suportado: {fmt}")

        return normalize_pyg_data(data, self.device)

    @staticmethod
    def _from_torch_object(obj: Any) -> Data:
        """Convert a Torch-loaded object to PyG Data."""
        if isinstance(obj, Data):
            return obj
        if isinstance(obj, dict):
            return Data(
                x=obj.get("x"),
                edge_index=obj["edge_index"],
                y=obj.get("y"),
                num_nodes=obj.get("num_nodes"),
            )
        raise TypeError("Arquivo .pt precisa conter torch_geometric.data.Data ou dict compatível.")

    @staticmethod
    def _from_npz(path: Path) -> Data:
        """Load a PyG graph from an NPZ file."""
        arrays = np.load(path, allow_pickle=True)
        edge_index = torch.as_tensor(arrays["edge_index"], dtype=torch.long)
        x = torch.as_tensor(arrays["x"], dtype=torch.float32) if "x" in arrays else None
        y = torch.as_tensor(arrays["y"], dtype=torch.long) if "y" in arrays else None
        num_nodes = int(arrays["num_nodes"]) if "num_nodes" in arrays else None
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    @staticmethod
    def _from_edge_csv(path: Path) -> Data:
        """Load an edge-list CSV.

        The CSV must contain ``source`` and ``target`` columns. For supervised
        node classification, labels still need to be added in a .pt/.npz file;
        this option is mainly useful for quick structural experiments.
        """
        df = pd.read_csv(path)
        if not {"source", "target"}.issubset(df.columns):
            raise ValueError("CSV precisa conter colunas `source` e `target`.")
        edge_index = torch.as_tensor(df[["source", "target"]].to_numpy().T, dtype=torch.long)
        num_nodes = int(edge_index.max().item() + 1)
        y = torch.as_tensor(df["label"].to_numpy(), dtype=torch.long) if "label" in df.columns else None
        return Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
