#!/usr/bin/env python
"""
Script to compare quality of depth predictions between different runs.
Useful for evaluating optimization impact on quality.
"""

import numpy as np
import argparse
import os
from pathlib import Path

def load_and_clean_predictions(path):
    """Load predictions and handle inf/nan values."""
    data = np.load(path)
    # Replace inf/nan for analysis
    data_clean = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return data, data_clean

def compare_predictions(original_path, optimized_path, verbose=True):
    """Compare two depth prediction files."""
    # Load predictions
    orig, orig_clean = load_and_clean_predictions(original_path)
    opt, opt_clean = load_and_clean_predictions(optimized_path)

    if verbose:
        print(f'Original shape: {orig.shape}, dtype: {orig.dtype}')
        print(f'Optimized shape: {opt.shape}, dtype: {opt.dtype}')

    # Check for inf/nan
    orig_has_inf = np.isinf(orig).any()
    orig_has_nan = np.isnan(orig).any()
    opt_has_inf = np.isinf(opt).any()
    opt_has_nan = np.isnan(opt).any()

    if verbose:
        print(f'\nOriginal - has inf: {orig_has_inf}, has nan: {orig_has_nan}')
        print(f'Optimized - has inf: {opt_has_inf}, has nan: {opt_has_nan}')

    # Ensure same shape for comparison
    min_frames = min(orig.shape[0], opt.shape[0])
    orig_compare = orig_clean[:min_frames]
    opt_compare = opt_clean[:min_frames]

    # Calculate statistics
    results = {}

    # Basic stats
    results['orig_stats'] = {
        'min': float(orig_compare.min()),
        'max': float(orig_compare.max()),
        'mean': float(orig_compare.mean()),
        'std': float(orig_compare.std())
    }

    results['opt_stats'] = {
        'min': float(opt_compare.min()),
        'max': float(opt_compare.max()),
        'mean': float(opt_compare.mean()),
        'std': float(opt_compare.std())
    }

    # Difference metrics
    diff = orig_compare - opt_compare
    abs_diff = np.abs(diff)

    results['diff_metrics'] = {
        'mae': float(abs_diff.mean()),  # Mean Absolute Error
        'max_error': float(abs_diff.max()),
        'rmse': float(np.sqrt((diff**2).mean())),  # Root Mean Square Error
        'relative_error_pct': float(abs_diff.mean() / np.abs(orig_compare).mean() * 100)
    }

    # Percentile differences
    results['percentile_diffs'] = {}
    for p in [50, 90, 95, 99]:
        results['percentile_diffs'][f'p{p}'] = float(np.percentile(abs_diff, p))

    # Correlation (if scipy available)
    try:
        from scipy import stats
        correlation = stats.pearsonr(orig_compare.flatten(), opt_compare.flatten())[0]
        results['correlation'] = float(correlation)
    except ImportError:
        results['correlation'] = None

    if verbose:
        print('\n--- Quality Comparison ---')
        print(f"Original - min: {results['orig_stats']['min']:.3f} "
              f"max: {results['orig_stats']['max']:.3f} "
              f"mean: {results['orig_stats']['mean']:.3f} "
              f"std: {results['orig_stats']['std']:.3f}")
        print(f"Optimized - min: {results['opt_stats']['min']:.3f} "
              f"max: {results['opt_stats']['max']:.3f} "
              f"mean: {results['opt_stats']['mean']:.3f} "
              f"std: {results['opt_stats']['std']:.3f}")

        print('\n--- Difference Statistics ---')
        print(f"Mean Absolute Error (MAE): {results['diff_metrics']['mae']:.4f}")
        print(f"Max Absolute Error: {results['diff_metrics']['max_error']:.4f}")
        print(f"Root Mean Square Error (RMSE): {results['diff_metrics']['rmse']:.4f}")
        print(f"Relative Error (%): {results['diff_metrics']['relative_error_pct']:.2f}%")

        if results['correlation'] is not None:
            print(f"Pearson Correlation: {results['correlation']:.4f}")

        print('\n--- Percentile Differences ---')
        for p, val in results['percentile_diffs'].items():
            print(f"{p}: {val:.4f}")

    return results

def visualize_comparison(original_path, optimized_path, output_path=None):
    """Create visual comparison of predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Load data
    orig, orig_clean = load_and_clean_predictions(original_path)
    opt, opt_clean = load_and_clean_predictions(optimized_path)

    # Use first frame for visualization
    orig_frame = orig_clean[0] if len(orig_clean) > 0 else None
    opt_frame = opt_clean[0] if len(opt_clean) > 0 else None

    if orig_frame is None or opt_frame is None:
        print("No data to visualize")
        return

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Original
    axes[0, 0].imshow(orig_frame, cmap='viridis')
    axes[0, 0].set_title('Original (2000 iterations)')
    axes[0, 0].axis('off')

    axes[0, 1].hist(orig_frame.flatten(), bins=50, alpha=0.7)
    axes[0, 1].set_title('Original - Value Distribution')
    axes[0, 1].set_xlabel('Depth Value')
    axes[0, 1].set_ylabel('Frequency')

    # Optimized
    axes[1, 0].imshow(opt_frame, cmap='viridis')
    axes[1, 0].set_title('Optimized (~530 iterations)')
    axes[1, 0].axis('off')

    axes[1, 1].hist(opt_frame.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Optimized - Value Distribution')
    axes[1, 1].set_xlabel('Depth Value')
    axes[1, 1].set_ylabel('Frequency')

    # Difference
    diff = orig_frame - opt_frame
    im = axes[0, 2].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('Difference (Original - Optimized)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # Absolute difference
    abs_diff = np.abs(diff)
    im2 = axes[1, 2].imshow(abs_diff, cmap='hot', vmin=0, vmax=0.2)
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare depth prediction quality')
    parser.add_argument('original', help='Path to original predictions (.npy)')
    parser.add_argument('optimized', help='Path to optimized predictions (.npy)')
    parser.add_argument('--visualize', action='store_true', help='Create visual comparison')
    parser.add_argument('--output', help='Output path for visualization')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Compare predictions
    results = compare_predictions(
        args.original,
        args.optimized,
        verbose=not args.quiet
    )

    # Visualize if requested
    if args.visualize:
        visualize_comparison(
            args.original,
            args.optimized,
            args.output
        )

    return results

if __name__ == '__main__':
    main()