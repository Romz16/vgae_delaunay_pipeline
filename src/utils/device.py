"""Device selection utilities."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string into a PyTorch device.

    Args:
        device: ``auto``, ``cpu``, ``cuda`` or a concrete PyTorch device string.

    Returns:
        A ``torch.device`` instance.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
