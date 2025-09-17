#!/usr/bin/env python
"""
Script to benchmark RollingDepth performance with different settings.
Useful for testing optimization impact.
"""

import argparse
import time
import subprocess
import json
from pathlib import Path
import sys

def run_benchmark(video_path, output_dir, checkpoint, preset='fast',
                 res=384, frames=10, dtype='fp16', verbose=False):
    """Run a single benchmark."""

    cmd = [
        sys.executable, 'run_video.py',
        '-i', str(video_path),
        '-o', str(output_dir),
        '-c', str(checkpoint),
        '--preset', preset,
        '--res', str(res),
        '--frame-count', str(frames),
        '--dtype', dtype
    ]

    if verbose:
        cmd.append('--verbose')

    start_time = time.time()

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    end_time = time.time()
    total_time = end_time - start_time

    # Parse timing from output if verbose
    timings = {}
    if verbose and result.stderr:
        lines = result.stderr.split('\n')
        for line in lines:
            if 'VAE encoding took' in line:
                timings['vae_encoding'] = float(line.split('took')[1].split('s')[0].strip())
            elif 'Initial inference took' in line:
                timings['inference'] = float(line.split('took')[1].split('s')[0].strip())
            elif 'Co-alignment took' in line and 'optimization' not in line:
                timings['coalignment'] = float(line.split('took')[1].split('s')[0].strip())
            elif 'Early stopping at iteration' in line:
                # Extract iteration number
                timings['early_stop_iter'] = int(line.split('iteration')[1].split('(')[0].strip())

    return {
        'total_time': total_time,
        'timings': timings,
        'settings': {
            'video': str(video_path),
            'resolution': res,
            'frames': frames,
            'dtype': dtype,
            'preset': preset
        }
    }

def compare_benchmarks(configs, video_path, checkpoint, output_base='output/benchmark'):
    """Run multiple benchmark configurations and compare."""

    results = []
    output_base = Path(output_base)

    for i, config in enumerate(configs):
        print(f"\n--- Running benchmark {i+1}/{len(configs)} ---")
        print(f"Config: {config}")

        output_dir = output_base / f"run_{i:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_benchmark(
            video_path=video_path,
            output_dir=output_dir,
            checkpoint=checkpoint,
            **config
        )

        result['config_name'] = config.get('name', f'Config {i+1}')
        results.append(result)

        print(f"Total time: {result['total_time']:.2f}s")
        if result['timings']:
            for key, val in result['timings'].items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.2f}s")
                else:
                    print(f"  {key}: {val}")

    return results

def print_comparison(results):
    """Print comparison of benchmark results."""

    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)

    # Find baseline (first result)
    baseline = results[0]
    baseline_time = baseline['total_time']

    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Resolution: {result['settings']['resolution']}px")
        print(f"  Frames: {result['settings']['frames']}")
        print(f"  Dtype: {result['settings']['dtype']}")
        print(f"  Total time: {result['total_time']:.2f}s", end='')

        if result != baseline:
            speedup = baseline_time / result['total_time']
            print(f" ({speedup:.2f}x vs baseline)")
        else:
            print(" (baseline)")

        if result['timings']:
            print("  Breakdown:")
            for key, val in result['timings'].items():
                if isinstance(val, float):
                    print(f"    {key}: {val:.2f}s")
                else:
                    print(f"    {key}: {val}")

def save_results(results, output_path):
    """Save benchmark results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark RollingDepth performance')
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-c', '--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--configs', help='JSON file with benchmark configurations')
    parser.add_argument('--output', default='output/benchmark', help='Output directory')
    parser.add_argument('--save-json', help='Save results to JSON file')

    args = parser.parse_args()

    # Default configurations if none provided
    if args.configs:
        with open(args.configs) as f:
            configs = json.load(f)
    else:
        configs = [
            {
                'name': 'Original (2000 iter)',
                'res': 384,
                'frames': 10,
                'dtype': 'fp16',
                'verbose': True
            },
            {
                'name': 'FP32 vs FP16',
                'res': 384,
                'frames': 10,
                'dtype': 'fp32',
                'verbose': True
            },
            {
                'name': 'Higher Resolution',
                'res': 512,
                'frames': 10,
                'dtype': 'fp16',
                'verbose': True
            },
            {
                'name': 'More Frames',
                'res': 384,
                'frames': 20,
                'dtype': 'fp16',
                'verbose': True
            }
        ]

    # Run benchmarks
    results = compare_benchmarks(
        configs=configs,
        video_path=args.input,
        checkpoint=args.checkpoint,
        output_base=args.output
    )

    # Print comparison
    print_comparison(results)

    # Save if requested
    if args.save_json:
        save_results(results, args.save_json)

if __name__ == '__main__':
    main()