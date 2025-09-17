#!/usr/bin/env python
"""
Comprehensive fix for depth_aligner to handle overlapping snippets properly.

The main issue: With overlapping snippets (stride=1), we need to:
1. Know the actual frame indices for each snippet
2. Properly merge overlapping predictions
3. Return the correct number of frames
"""

import torch
import numpy as np
from typing import List, Tuple

def simulate_get_snippet_indices(seq_len: int, snippet_len: int, stride: int = 1, dilation: int = 1) -> List[List[int]]:
    """Simulate how the pipeline generates snippet indices"""
    gap = dilation - 1
    snippet_idx_ls = []

    total_window_size = (snippet_len - 1) * (gap + 1) + 1
    # index of the first frame
    i_start_ls = list(range(0, seq_len - total_window_size + 1, stride))
    # last window (for stride > 1)
    if len(i_start_ls) > 0 and i_start_ls[-1] < seq_len - total_window_size:
        i_start_ls.append(seq_len - total_window_size)

    for i_start in i_start_ls:
        input_idx = list(range(i_start, i_start + total_window_size, gap + 1))
        snippet_idx_ls.append(input_idx)

    return snippet_idx_ls

def merge_overlapping_snippets(
    snippet_data: List[torch.Tensor],
    snippet_indices: List[List[int]],
    seq_len: int
) -> torch.Tensor:
    """
    Merge overlapping snippet predictions.

    Args:
        snippet_data: List of tensors, each shape (num_snippets, snippet_len, C, H, W)
        snippet_indices: List of frame indices for each snippet level
        seq_len: Total number of frames expected

    Returns:
        Merged tensor of shape (seq_len, C, H, W)
    """

    # Get dimensions from first snippet
    if not snippet_data or len(snippet_data[0]) == 0:
        raise ValueError("No snippet data provided")

    # Assuming all snippets have same spatial dimensions
    first_snippet = snippet_data[0][0]  # First snippet of first dilation level
    C, H, W = first_snippet.shape[1], first_snippet.shape[2], first_snippet.shape[3]
    device = first_snippet.device
    dtype = first_snippet.dtype

    # Initialize output and count tensors
    merged_frames = torch.zeros((seq_len, C, H, W), device=device, dtype=dtype)
    frame_counts = torch.zeros((seq_len,), device=device, dtype=torch.int32)

    # Process each dilation level
    for dilation_idx, (snippets, indices_list) in enumerate(zip(snippet_data, snippet_indices)):
        # snippets: shape (num_snippets, snippet_len, C, H, W)
        # indices_list: list of frame indices for each snippet

        for snippet_idx, frame_indices in enumerate(indices_list):
            for pos_in_snippet, frame_idx in enumerate(frame_indices):
                if frame_idx < seq_len:  # Safety check
                    # Add this snippet's prediction for this frame
                    merged_frames[frame_idx] += snippets[snippet_idx, pos_in_snippet]
                    frame_counts[frame_idx] += 1

    # Average where we have multiple predictions
    for frame_idx in range(seq_len):
        if frame_counts[frame_idx] > 0:
            merged_frames[frame_idx] /= frame_counts[frame_idx].float()
        else:
            print(f"WARNING: Frame {frame_idx} has no predictions!")

    return merged_frames

def test_merge():
    """Test the merging function"""

    # Test case: 5 frames, snippet_len=3, stride=1
    seq_len = 5
    snippet_len = 3
    stride = 1
    C, H, W = 1, 4, 4  # Small dims for testing

    # Generate snippet indices
    snippet_indices = [simulate_get_snippet_indices(seq_len, snippet_len, stride)]
    print(f"Snippet indices: {snippet_indices[0]}")

    # Create dummy snippet data
    # 3 snippets, each with 3 frames
    num_snippets = len(snippet_indices[0])
    snippet_data = []

    snippets = torch.zeros((num_snippets, snippet_len, C, H, W))
    for i in range(num_snippets):
        # Mark each snippet with unique values for testing
        snippets[i] = (i + 1) * torch.ones((snippet_len, C, H, W))

    snippet_data.append(snippets)

    print(f"\nSnippet data shape: {snippets.shape}")
    print(f"Snippet 0 (frames 0,1,2): value={snippets[0,0,0,0,0].item()}")
    print(f"Snippet 1 (frames 1,2,3): value={snippets[1,0,0,0,0].item()}")
    print(f"Snippet 2 (frames 2,3,4): value={snippets[2,0,0,0,0].item()}")

    # Test merge
    merged = merge_overlapping_snippets(snippet_data, snippet_indices, seq_len)

    print(f"\nMerged shape: {merged.shape}")
    print(f"Expected shape: ({seq_len}, {C}, {H}, {W})")

    # Check merged values
    print("\nMerged frame values (should be averaged where overlapping):")
    for frame_idx in range(seq_len):
        value = merged[frame_idx, 0, 0, 0].item()
        print(f"  Frame {frame_idx}: {value:.2f}")

    # Expected values:
    # Frame 0: Only in snippet 0 (value 1) -> 1.0
    # Frame 1: In snippets 0,1 (values 1,2) -> 1.5
    # Frame 2: In snippets 0,1,2 (values 1,2,3) -> 2.0
    # Frame 3: In snippets 1,2 (values 2,3) -> 2.5
    # Frame 4: Only in snippet 2 (value 3) -> 3.0

    expected = [1.0, 1.5, 2.0, 2.5, 3.0]
    print("\nExpected values:", expected)

    # Verify
    for i, exp in enumerate(expected):
        actual = merged[i, 0, 0, 0].item()
        assert abs(actual - exp) < 0.01, f"Frame {i}: expected {exp}, got {actual}"

    print("\n✓ Test passed! Merging works correctly.")

if __name__ == "__main__":
    test_merge()