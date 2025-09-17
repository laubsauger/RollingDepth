#!/usr/bin/env python
import torch
import sys

# Test to understand the depth_aligner issue

def simulate_snippet_generation(seq_len=5, snippet_len=3, stride=1, dilation=1):
    """Simulate how snippets are generated in the pipeline"""
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

def test_depth_aligner_logic():
    """Test the depth aligner logic with overlapping snippets"""

    # Parameters matching the actual case
    seq_len = 5
    snippet_len = 3
    stride = 1
    dilation = 1

    # Generate snippets
    snippet_indices = simulate_snippet_generation(seq_len, snippet_len, stride, dilation)
    print(f"Input: {seq_len} frames, snippet_len={snippet_len}, stride={stride}, dilation={dilation}")
    print(f"Generated snippets: {snippet_indices}")
    print(f"Number of snippets: {len(snippet_indices)}")

    # Simulate snippet data (batch_size, snippet_length, H, W)
    # Each snippet has snippet_len frames
    H, W = 10, 10  # Small size for testing
    snippet_data = []
    for i, indices in enumerate(snippet_indices):
        # Create dummy data for this snippet
        data = torch.ones(1, snippet_len, 1, H, W) * (i + 1)  # Mark with snippet number
        snippet_data.append(data)

    print(f"\nSnippet data shapes:")
    for i, data in enumerate(snippet_data):
        print(f"  Snippet {i}: shape {data.shape}, indices {snippet_indices[i]}")

    # Now test the depth_aligner's create_triplet_indices logic
    print(f"\nDepth aligner's perspective:")

    # The issue: depth_aligner assumes it's getting the TOTAL sequence length
    # But it's actually getting overlapping snippets

    # What depth_aligner calculates (incorrectly):
    gap = dilation - 1  # = 0
    num_snippets = len(snippet_data[0])  # This is the number of snippets
    snippet_length = snippet_data[0].shape[1]  # This is snippet_len

    print(f"  num_snippets (from data): {num_snippets}")
    print(f"  snippet_length: {snippet_length}")

    # The wrong calculation:
    # sequence_length_wrong = num_snippets * snippet_length - num_snippets + 1
    # print(f"  Wrong sequence_length: {sequence_length_wrong}")

    # The correct calculation for overlapping snippets with stride=1:
    sequence_length_correct = (len(snippet_indices) - 1) * stride + snippet_length
    print(f"  Correct sequence_length: {sequence_length_correct}")

    # Show how frames map to snippets
    print(f"\nFrame to snippet mapping:")
    for frame_idx in range(seq_len):
        containing_snippets = []
        for snippet_idx, indices in enumerate(snippet_indices):
            if frame_idx in indices:
                pos_in_snippet = indices.index(frame_idx)
                containing_snippets.append((snippet_idx, pos_in_snippet))
        print(f"  Frame {frame_idx}: appears in snippets {containing_snippets}")

    return snippet_indices, snippet_data

if __name__ == "__main__":
    test_depth_aligner_logic()