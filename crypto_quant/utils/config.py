"""Configuration utilities for API keys and reproducibility.

This module centralizes environment variables and common configuration
used across the project.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class APIKeys:
    """Holds API keys loaded from environment variables."""

    glassnode: str | None
    fred: str | None


def get_api_keys() -> APIKeys:
    """Load API keys from environment variables.

    Returns:
        APIKeys: Dataclass with optional keys.
    """

    return APIKeys(
        glassnode=os.getenv("GLASSNODE_API_KEY"),
        fred=os.getenv("FRED_API_KEY"),
    )


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility where possible.

    Args:
        seed (int): Seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # PyTorch is optional at import-time in many modules
        pass


def project_root() -> str:
    """Return absolute path to project root (directory containing this package)."""

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
