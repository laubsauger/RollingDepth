# RollingDepth - Quick Start Guide

**Optimized version with Apple Silicon support and 2x faster processing**

## Installation

```bash
# Clone repository
git clone <this-repository>
cd RollingDepth

# Setup Python environment (3.11 for TouchDesigner)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Usage

### Fastest Processing (Recommended)
```bash
python run_video.py -i input.mp4 -o output/result -c prs-eth/rollingdepth-v1-0 \
    --quality fast --preset fast --res 384 --dtype fp16
```

### Standard Quality
```bash
python run_video.py -i input.mp4 -o output/result -c prs-eth/rollingdepth-v1-0
```

### Maximum Quality
```bash
python run_video.py -i input.mp4 -o output/result -c prs-eth/rollingdepth-v1-0 \
    --quality quality --preset full --res 768
```

## Key Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--quality` | fast, balanced, quality | balanced | Speed/quality trade-off |
| `--res` | 256, 384, 512, 768 | 768 | Processing resolution |
| `--dtype` | fp16, fp32 | fp16 | Precision (fp16 is faster) |
| `--preset` | fast, full | fast | Inference preset |

## Performance Guide

| Settings | Time (20 frames) | Use Case |
|----------|------------------|----------|
| `--quality fast --res 384 --dtype fp16` | ~23s | Real-time, previews |
| `--quality fast --res 512 --dtype fp16` | ~37s | Higher quality preview |
| `--quality quality --res 768` | ~70s+ | Final renders |

## Output Files

- `*_pred.npy` - Raw depth data
- `*_Spectral_r.mp4` - Colored depth video
- `*_rgbd.mp4` - Side-by-side RGB and depth

## TouchDesigner Integration

```python
# In TouchDesigner Python node
import subprocess

cmd = [
    "python", "run_video.py",
    "-i", "input.mp4",
    "-o", "output",
    "-c", "checkpoint/rollingdepth-v1-0",
    "--quality", "fast",
    "--res", "384",
    "--dtype", "fp16"
]

subprocess.run(cmd)
```

## Tips

- Use `--quality fast` for 2x speedup with minimal quality loss (1.25% error)
- Use `--res 384` for best speed/quality balance
- Use `--dtype fp16` on Apple Silicon for better performance
- Process in chunks with `--frame-count 20` for consistent memory usage

---
*For technical details and benchmarks, see README_TECHNICAL.md*