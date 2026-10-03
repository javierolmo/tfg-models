"""Pytest root configuration and runtime pre-initialization."""

# Pre-initialize PyTorch runtime before C-extensions / OpenMP runtimes to avoid library clashes
try:
    import torch  # noqa: F401
except ImportError:
    pass
