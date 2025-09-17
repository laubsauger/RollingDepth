#!/usr/bin/env python3

import torch
import numpy as np
import einops
from rollingdepth import concatenate_videos_horizontally_torch, write_video_from_numpy

# Create test tensors similar to what we have
N, H, W = 3, 432, 768

# Create a simple RGB tensor (grayscale gradient)
rgb = torch.zeros(N, 3, H, W, dtype=torch.float32)
for i in range(N):
    # Create a gradient pattern
    for c in range(3):
        rgb[i, c] = torch.linspace(0, 255, H).unsqueeze(1).repeat(1, W)

# Create a simple depth tensor (different gradient)
depth = torch.zeros(N, 3, H, W, dtype=torch.float32)
for i in range(N):
    for c in range(3):
        depth[i, c] = torch.linspace(255, 0, W).unsqueeze(0).repeat(H, 1)

print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.1f}, {rgb.max():.1f}]")
print(f"Depth shape: {depth.shape}, range: [{depth.min():.1f}, {depth.max():.1f}]")

# Test concatenation
concat = concatenate_videos_horizontally_torch(rgb, depth, gap=10)
print(f"Concatenated shape: {concat.shape}, range: [{concat.min():.1f}, {concat.max():.1f}]")

# Convert to video format
concat_np = torch.clamp(concat, 0, 255).byte().cpu().numpy()
concat_np = einops.rearrange(concat_np, "n c h w -> n h w c")
print(f"Final numpy shape: {concat_np.shape}, dtype: {concat_np.dtype}, range: [{concat_np.min()}, {concat_np.max()}]")

# Write test video
write_video_from_numpy(
    frames=concat_np,
    output_path="output/test_concat_debug.mp4",
    fps=10,
    crf=23,
    preset="medium",
    verbose=True,
)
print("Test video written to output/test_concat_debug.mp4")