#!/usr/bin/env python
"""
Compare different num_view (snippet_lengths) settings.
"""

import numpy as np
from pathlib import Path

def main():
    print("=" * 70)
    print("NUM_VIEW (SNIPPET_LENGTHS) COMPARISON")
    print("=" * 70)

    # Results from our tests
    configs = [
        {
            "name": "snippet_lengths=3 (default)",
            "path": "output/horse_diffusers_0.35_fast/2_horse_pred.npy",
            "frames": 20,
            "initial_time": 19.66,
            "coalign_time": 10.70,
            "total_time": 30.36,
            "iterations": 531
        },
        {
            "name": "snippet_lengths=4",
            "path": "output/test_snippet_4/2_horse_pred.npy",
            "frames": 12,
            "initial_time": 11.51,
            "coalign_time": 4.76,
            "total_time": 16.27,
            "iterations": 531
        }
    ]

    print("\n### PERFORMANCE COMPARISON")
    print("-" * 50)

    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Frames: {config['frames']}")
        print(f"  Initial inference: {config['initial_time']:.2f}s")
        print(f"  Co-alignment: {config['coalign_time']:.2f}s ({config['iterations']} iter)")
        print(f"  Total: {config['total_time']:.2f}s")
        print(f"  Per frame: {config['total_time']/config['frames']:.2f}s")

        if Path(config['path']).exists():
            pred = np.load(config['path'])
            print(f"  Output shape: {pred.shape}")

    print("\n" + "=" * 70)
    print("ANALYSIS: What num_view Does")
    print("=" * 70)

    print("""
NUM_VIEW determines how many frames can "see" each other during attention:

1. TEMPORAL CONTEXT WINDOW
   - num_view=3: Each pixel attends to 3×1296 = 3,888 positions
   - num_view=4: Each pixel attends to 4×1296 = 5,184 positions
   - num_view=5: Would attend to 5×1296 = 6,480 positions

2. ATTENTION MATRIX SIZE (at 384px resolution)
   - num_view=3: 3,888 × 3,888 = 15.1M attention values
   - num_view=4: 5,184 × 5,184 = 26.9M attention values (1.8× larger)
   - num_view=5: 6,480 × 6,480 = 42.0M attention values (2.8× larger)

3. MEMORY SCALING
   - Memory usage grows quadratically: O(num_view²)
   - Computation time also grows quadratically

4. QUALITY VS PERFORMANCE TRADE-OFF
   - Higher num_view = Better temporal consistency
   - Higher num_view = More memory and slower processing
   - Model was trained with num_view=3 as default
""")

    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
1. DEFAULT (num_view=3): Best for most cases
   - Optimal balance of quality and speed
   - What the model was trained with

2. HIGHER (num_view=4): Consider for:
   - Scenes with fast motion
   - When temporal consistency is critical
   - When you have extra GPU memory

3. LOWER (num_view=2): Consider for:
   - Real-time applications
   - Memory-constrained systems
   - Static scenes

4. EXPERIMENTAL IDEAS:
   - Dynamic num_view based on motion detection
   - Different num_view for different pyramid levels
   - Adaptive based on available memory

NOTE: The model likely performs best at num_view=3 since that's
what it was trained with. Deviations might reduce quality.
""")

if __name__ == "__main__":
    main()