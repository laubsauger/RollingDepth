#!/usr/bin/env python
"""
Compare the three quality modes: fast, balanced, and quality.
Analyzes performance vs quality trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess
import time
from pathlib import Path
import sys
import os

def run_with_quality_mode(video_path, output_dir, checkpoint, quality_mode,
                         res=384, frames=10, dtype='fp16', verbose=True):
    """Run RollingDepth with specific quality mode."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'run_video.py',
        '-i', str(video_path),
        '-o', str(output_dir),
        '-c', str(checkpoint),
        '--preset', 'fast',
        '--res', str(res),
        '--frame-count', str(frames),
        '--dtype', dtype,
        '--quality', quality_mode
    ]

    if verbose:
        cmd.append('--verbose')

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    # Parse results
    info = {
        'mode': quality_mode,
        'total_time': end_time - start_time,
        'iterations': None,
        'coalignment_time': None
    }

    if result.stderr:
        lines = result.stderr.split('\n')
        for line in lines:
            if 'Early stopping at iteration' in line:
                info['iterations'] = int(line.split('iteration')[1].split('(')[0].strip())
            elif 'Co-alignment took' in line and 'optimization' not in line:
                info['coalignment_time'] = float(line.split('took')[1].split('s')[0].strip())

    # If no early stopping, assume full 2000 iterations
    if info['iterations'] is None:
        info['iterations'] = 2000

    return info

def compare_predictions(pred1_path, pred2_path):
    """Quick quality comparison."""
    pred1 = np.load(pred1_path).astype(np.float32)
    pred2 = np.load(pred2_path).astype(np.float32)

    # Handle inf/nan
    pred1 = np.nan_to_num(pred1, nan=0.0, posinf=1.0, neginf=-1.0)
    pred2 = np.nan_to_num(pred2, nan=0.0, posinf=1.0, neginf=-1.0)

    # Ensure same shape
    min_frames = min(pred1.shape[0], pred2.shape[0])
    pred1 = pred1[:min_frames]
    pred2 = pred2[:min_frames]

    # Calculate metrics
    diff = pred1 - pred2
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff**2).mean())

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'relative_error': float(mae / np.abs(pred1).mean() * 100)
    }

def create_comparison_visualization(results, output_path):
    """Create visualization comparing the three modes."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    modes = [r['mode'] for r in results]
    times = [r['total_time'] for r in results]
    iterations = [r['iterations'] for r in results]
    coalign_times = [r.get('coalignment_time', 0) for r in results]

    # Color scheme
    colors = {'fast': '#ff7f0e', 'balanced': '#2ca02c', 'quality': '#1f77b4'}
    mode_colors = [colors.get(m, '#333333') for m in modes]

    # 1. Total time comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(modes, times, color=mode_colors, alpha=0.7)
    ax1.set_ylabel('Total Time (s)')
    ax1.set_title('Processing Time by Quality Mode')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center')

    # 2. Iterations used
    ax2 = axes[0, 1]
    bars2 = ax2.bar(modes, iterations, color=mode_colors, alpha=0.7)
    ax2.set_ylabel('Iterations')
    ax2.set_title('Co-alignment Iterations')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='Maximum')

    for bar, val in zip(bars2, iterations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(val), ha='center')

    # 3. Co-alignment time
    ax3 = axes[0, 2]
    bars3 = ax3.bar(modes, coalign_times, color=mode_colors, alpha=0.7)
    ax3.set_ylabel('Co-alignment Time (s)')
    ax3.set_title('Co-alignment Optimization Time')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars3, coalign_times):
        if val:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}s', ha='center')

    # 4. Efficiency (iterations per second)
    ax4 = axes[1, 0]
    efficiency = [it/ct if ct else 0 for it, ct in zip(iterations, coalign_times)]
    bars4 = ax4.bar(modes, efficiency, color=mode_colors, alpha=0.7)
    ax4.set_ylabel('Iterations per Second')
    ax4.set_title('Optimization Efficiency')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Quality vs baseline
    ax5 = axes[1, 1]
    if 'quality_metrics' in results[0]:
        mae_values = [r['quality_metrics'].get('mae', 0) for r in results]
        bars5 = ax5.bar(modes, mae_values, color=mode_colors, alpha=0.7)
        ax5.set_ylabel('Mean Absolute Error')
        ax5.set_title('Quality Comparison (Lower is Better)')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars5, mae_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center')

    # 6. Speed vs Quality trade-off
    ax6 = axes[1, 2]
    if 'quality_metrics' in results[0]:
        speedups = [results[-1]['total_time'] / r['total_time'] for r in results]
        errors = [r['quality_metrics'].get('relative_error', 0) for r in results]

        ax6.scatter(speedups, errors, s=200, c=mode_colors, alpha=0.7)

        for x, y, mode in zip(speedups, errors, modes):
            ax6.annotate(mode, (x, y), fontsize=12, fontweight='bold',
                        ha='center', va='center')

        ax6.set_xlabel('Speedup vs Quality Mode')
        ax6.set_ylabel('Relative Error (%)')
        ax6.set_title('Speed-Quality Trade-off')
        ax6.grid(True, alpha=0.3)

        # Add ideal direction arrow
        ax6.annotate('', xy=(2.5, 0), xytext=(1, 5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
        ax6.text(1.8, 2, 'Better', color='green', fontweight='bold', alpha=0.7)

    plt.suptitle('RollingDepth Quality Modes Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare RollingDepth quality modes')
    parser.add_argument('-i', '--input', required=True, help='Input video')
    parser.add_argument('-c', '--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--run-tests', action='store_true', help='Run the inference tests')
    parser.add_argument('--output-base', default='output/quality_comparison', help='Output directory')

    args = parser.parse_args()

    results = []

    if args.run_tests:
        print("Running quality mode comparisons...")

        for mode in ['fast', 'balanced', 'quality']:
            print(f"\n--- Testing {mode} mode ---")
            output_dir = Path(args.output_base) / mode

            info = run_with_quality_mode(
                video_path=args.input,
                output_dir=output_dir,
                checkpoint=args.checkpoint,
                quality_mode=mode
            )

            print(f"  Time: {info['total_time']:.2f}s")
            print(f"  Iterations: {info['iterations']}")
            print(f"  Co-alignment: {info['coalignment_time']:.2f}s")

            results.append(info)

    else:
        # Load existing results
        print("Using existing results from output/quality_test/")

        for mode in ['fast', 'balanced', 'quality']:
            output_dir = Path('output/quality_test') / mode
            pred_path = output_dir / '1_gokart_cropped_pred.npy'

            if pred_path.exists():
                info = {'mode': mode}

                # Estimate timing based on our tests
                if mode == 'fast':
                    info['total_time'] = 12.0
                    info['iterations'] = 559
                    info['coalignment_time'] = 4.63
                elif mode == 'balanced':
                    info['total_time'] = 24.0
                    info['iterations'] = 2000
                    info['coalignment_time'] = 17.22
                else:  # quality
                    info['total_time'] = 23.5
                    info['iterations'] = 2000
                    info['coalignment_time'] = 16.61

                results.append(info)

    # Calculate quality metrics if predictions exist
    quality_base = Path('output/quality_test/quality/1_gokart_cropped_pred.npy')
    if quality_base.exists():
        for i, result in enumerate(results):
            mode = result['mode']
            pred_path = Path('output/quality_test') / mode / '1_gokart_cropped_pred.npy'

            if pred_path.exists() and mode != 'quality':
                metrics = compare_predictions(quality_base, pred_path)
                result['quality_metrics'] = metrics
            elif mode == 'quality':
                result['quality_metrics'] = {'mae': 0, 'rmse': 0, 'relative_error': 0}

    # Print summary
    print("\n" + "="*60)
    print("QUALITY MODES SUMMARY")
    print("="*60)

    for r in results:
        print(f"\n{r['mode'].upper()} Mode:")
        print(f"  Total time: {r['total_time']:.2f}s")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Co-alignment: {r.get('coalignment_time', 'N/A')}s")
        if 'quality_metrics' in r:
            print(f"  MAE vs quality: {r['quality_metrics']['mae']:.4f}")
            print(f"  Relative error: {r['quality_metrics']['relative_error']:.2f}%")

    # Create visualization
    output_path = Path(args.output_base) / 'quality_modes_comparison.png'
    create_comparison_visualization(results, output_path)

    # Save results
    json_path = Path(args.output_base) / 'quality_modes_results.json'
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

if __name__ == '__main__':
    main()