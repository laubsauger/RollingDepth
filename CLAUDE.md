# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RollingDepth is a video depth estimation model that processes videos through overlapping snippets (windows of frames) without requiring dedicated video models. It uses a modified diffusion architecture with cross-frame self-attention to achieve temporal consistency.

## Common Commands

### Setup and Dependencies
```bash
# Activate virtual environment (from project root)
source ../../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install modified diffusers (required for cross-frame attention)
pip install -e diffusers/
```

### Running Video Depth Estimation
```bash
# Basic inference with presets
python run_video.py -i data/samples/video.mp4 -o output/results -p fast

# With specific parameters
python run_video.py \
    -i data/samples/2_horse.mp4 \
    -o output/test \
    --frame-count 20 \
    --res 384 \
    --dtype fp16 \
    --quality fast
```

### Key Command Arguments
- `-p/--preset`: fast, fast1024, full, paper (controls speed/quality tradeoffs)
- `--quality`: fast, balanced, quality (controls co-alignment iterations)
- `--res`: Processing resolution (384, 512, 768, 1024)
- `--frame-count`: Number of frames to process
- `--dtype`: fp16 or fp32 precision
- `--save-sbs`: Save side-by-side RGB+depth video (default: True)
- `-c`: Checkpoint path (default: downloads from HuggingFace)

## Architecture and Key Concepts

### Cross-Frame Attention Mechanism
The model uses a custom UNet2D with cross-frame self-attention implemented in the diffusers library. Key parameter `num_view` controls how many frames can attend to each other (default=3).

**Location**: Modified in diffusers library - search for "Modified in RollingDepth" comments
**Memory scaling**: O(num_view²) - be cautious with larger values

### Overlapping Snippet Processing
With stride=1 and snippet_len=3, the pipeline processes:
- Snippet 0: frames [0,1,2]
- Snippet 1: frames [1,2,3]
- Snippet 2: frames [2,3,4]

This overlap ensures temporal consistency but requires proper handling in depth_aligner.

### Depth Aligner Module
**File**: `rollingdepth/depth_aligner.py`

Merges overlapping snippet predictions through optimization:
1. Accepts `seq_len` and `snippet_indices` to handle overlaps correctly
2. Optimizes scale/translation parameters to align snippets
3. Averages overlapping frame predictions in `merge_scaled_triplets()`

Key fix areas:
- Must pass correct `seq_len` from pipeline
- `merge_scaled_triplets()` must return shape (N, 1, H, W)

### Pipeline Architecture
**File**: `rollingdepth/rollingdepth_pipeline.py`

Main processing flow:
1. Load video frames and preprocess
2. Create overlapping snippets with specified dilations
3. Run diffusion inference on each snippet batch
4. Align snippets using DepthAligner
5. Optional refinement step
6. Export depth maps and videos

### Colorization
**File**: `src/util/colorize.py`

Handles both 3D (N,H,W) and 4D (N,1,H,W) depth tensors for visualization. Must squeeze 4D input before processing.

## Critical Implementation Details

### Shape Conventions
- Pipeline expects depth shape: (N, 1, H, W) where N = number of frames
- Colorize expects: (N, H, W) - requires squeeze if 4D
- Video export expects colored depth: (N, H, W, 3)

### Memory Management
- Use `--max-vae-bs 1` for low memory
- Use `--unload-snippet true` to free memory between snippets
- MPS devices benefit from periodic `torch.mps.synchronize()`

### Quality vs Speed Tradeoffs
- `quality` mode: 2000 iterations, strict convergence
- `balanced` mode: 1000-1200 iterations, moderate convergence
- `fast` mode: 500-700 iterations, aggressive early stopping

## Common Issues and Fixes

### Frame Count Mismatch
If RGB frames != depth frames, check:
1. Pipeline passes correct `seq_len` to depth_aligner
2. `snippet_indices` properly tracks frame mappings
3. `merge_scaled_triplets` handles all frames

### RGBD Video Corruption
Usually shape mismatch in colorization:
1. Check colorize receives correct shape (3D vs 4D)
2. Verify side-by-side concatenation dimensions match
3. Ensure consistent resolution/fps in video export

### Testing Individual Components
```python
# Test depth aligner fix
python test_depth_aligner.py

# Quick 5-frame test
python run_video.py -i data/samples/2_horse.mp4 -o output/test --frame-count 5
```

## File Structure Notes
- Core pipeline: `rollingdepth/`
- Utilities: `src/util/`
- Modified diffusers: `diffusers/` (contains cross-frame attention mods)
- Test scripts: Various `test_*.py` files in root