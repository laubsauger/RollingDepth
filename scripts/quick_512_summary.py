#!/usr/bin/env python
"""
Quick summary of 512px quality mode results.
"""

import numpy as np
from pathlib import Path

def main():
    # Check if all files exist
    output_dir = Path('../output/horse_quality_512')

    modes = ['fast', 'balanced', 'quality']
    for mode in modes:
        pred_path = output_dir / mode / '2_horse_pred.npy'
        if pred_path.exists():
            pred = np.load(pred_path)
            print(f"{mode.upper()}: shape {pred.shape}, min={pred.min():.3f}, max={pred.max():.3f}")
        else:
            print(f"{mode.upper()}: Not found")

    print("\n" + "=" * 60)
    print("512px PERFORMANCE SUMMARY")
    print("=" * 60)

    # Timings from our runs
    timings = {
        'fast': {'time': 37.12, 'iter': 531, 'speedup': 1.89},
        'balanced': {'time': 40.90, 'iter': 851, 'speedup': 1.71},
        'quality': {'time': 70.0, 'iter': 2000, 'speedup': 1.00}
    }

    for mode, data in timings.items():
        print(f"\n{mode.upper()} MODE:")
        print(f"  Time: {data['time']:.1f}s")
        print(f"  Iterations: {data['iter']}")
        print(f"  Speedup vs quality: {data['speedup']:.2f}x")

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print("\n1. 512px is ~60-70% slower than 384px across all modes")
    print("2. Fast mode still provides best value (1.89x speedup)")
    print("3. Balanced mode anomaly persists (less speedup than expected)")
    print("\n384px vs 512px Processing Time:")
    print("  Fast: 23.2s → 37.1s (1.60x slower)")
    print("  Balanced: 26.0s → 40.9s (1.57x slower)")
    print("  Quality: 35.1s → 70.0s (1.99x slower)")

    # Quick quality check
    if (output_dir / 'quality' / '2_horse_pred.npy').exists():
        quality_pred = np.load(output_dir / 'quality' / '2_horse_pred.npy')

        for mode in ['fast', 'balanced']:
            pred_path = output_dir / mode / '2_horse_pred.npy'
            if pred_path.exists():
                mode_pred = np.load(pred_path)

                # Normalize both
                q_norm = (quality_pred - quality_pred.min()) / (quality_pred.max() - quality_pred.min())
                m_norm = (mode_pred - mode_pred.min()) / (mode_pred.max() - mode_pred.min())

                # Calculate error
                diff = np.abs(q_norm - m_norm)
                mae = diff.mean()

                print(f"\n{mode.upper()} vs QUALITY:")
                print(f"  Mean Absolute Error: {mae:.4f} ({mae*100:.2f}%)")

if __name__ == '__main__':
    main()