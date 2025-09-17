"""Utility functions for MPS (Metal Performance Shaders) support on macOS."""

import torch
import logging


def clear_memory_cache(device=None):
    """Clear memory cache for the current device (CUDA, MPS, or CPU).

    Args:
        device: torch.device or None. If None, will detect from current tensors.
    """
    if device is None:
        # Try to detect device from current context
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS has its own cache clearing method
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    # CPU doesn't need cache clearing


def get_best_device():
    """Get the best available device for the current system.

    Returns:
        torch.device: Best available device (CUDA > MPS > CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check if MPS is available (Apple Silicon)
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_mps_available():
    """Check if MPS is available on the current system.

    Returns:
        bool: True if MPS is available, False otherwise
    """
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def convert_for_mps(tensor, dtype=None):
    """Convert tensor to MPS-compatible format if needed.

    Some operations on MPS require specific data types.

    Args:
        tensor: Input tensor
        dtype: Target dtype (default: None, auto-detect)

    Returns:
        Converted tensor
    """
    if tensor.device.type == "mps":
        # MPS doesn't support float64, convert to float32
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        # MPS doesn't support int64 for some operations, convert to int32
        elif tensor.dtype == torch.int64:
            # Check if this is an index tensor (preserve int64 for indexing)
            if not tensor.requires_grad:
                pass  # Keep int64 for indexing
            else:
                tensor = tensor.to(torch.int32)

    if dtype is not None:
        tensor = tensor.to(dtype)

    return tensor