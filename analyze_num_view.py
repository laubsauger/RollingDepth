#!/usr/bin/env python
"""
Deep analysis of num_view parameter and its effect on cross-frame attention.
This script explains and visualizes how num_view controls temporal consistency.
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

def visualize_attention_matrix(num_view, resolution=384):
    """Create a visualization of the attention matrix for different num_view values."""

    # Calculate sizes
    patch_size = 32  # Each patch is 32x32 pixels
    patches_per_frame = (resolution // patch_size) ** 2  # 144 for 384px
    total_patches = patches_per_frame * num_view

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Frame structure
    ax = axes[0]
    ax.set_title(f'Input: {num_view} Frames @ {resolution}px')
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution * num_view)

    # Draw frames
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    for i in range(num_view):
        y_start = i * resolution
        rect = FancyBboxPatch((10, y_start + 10), resolution - 20, resolution - 20,
                              boxstyle="round,pad=10",
                              facecolor=colors[i % len(colors)],
                              alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(resolution/2, y_start + resolution/2, f'Frame {i+1}',
               fontsize=16, ha='center', va='center', weight='bold')

    ax.set_aspect('equal')
    ax.axis('off')

    # Subplot 2: Attention matrix structure
    ax = axes[1]
    ax.set_title(f'Attention Matrix: {total_patches}×{total_patches}')
    ax.set_xlim(0, total_patches)
    ax.set_ylim(0, total_patches)

    # Draw attention blocks
    for i in range(num_view):
        for j in range(num_view):
            x_start = j * patches_per_frame
            y_start = i * patches_per_frame

            # Determine color based on attention type
            if i == j:
                # Self-attention within frame
                color = colors[i % len(colors)]
                alpha = 0.6
            else:
                # Cross-frame attention
                color = 'purple'
                alpha = 0.3

            rect = Rectangle((x_start, y_start), patches_per_frame, patches_per_frame,
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

    # Add labels
    for i in range(num_view):
        mid_point = (i + 0.5) * patches_per_frame
        ax.text(mid_point, -5, f'F{i+1}', fontsize=10, ha='center')
        ax.text(-5, mid_point, f'F{i+1}', fontsize=10, ha='center', rotation=90)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Subplot 3: Memory usage
    ax = axes[2]
    ax.set_title('Memory & Compute Scaling')

    num_views_range = [2, 3, 4, 5]
    memory_usage = [(n * patches_per_frame) ** 2 / 1e6 for n in num_views_range]

    bars = ax.bar(num_views_range, memory_usage, color=['gray' if n != num_view else colors[0] for n in num_views_range])
    ax.set_xlabel('num_view')
    ax.set_ylabel('Attention Values (millions)')
    ax.set_xticks(num_views_range)

    # Add values on bars
    for bar, val in zip(bars, memory_usage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}M', ha='center', va='bottom', fontsize=10)

    # Add legend for subplot 2
    legend_elements = [
        mpatches.Patch(color=colors[0], alpha=0.6, label='Within-frame attention'),
        mpatches.Patch(color='purple', alpha=0.3, label='Cross-frame attention')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.suptitle(f'Cross-Frame Attention with num_view={num_view}', fontsize=16, weight='bold')
    plt.tight_layout()

    # Save the figure
    output_dir = Path("output/num_view_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / f"attention_visualization_numview_{num_view}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return output_dir / f"attention_visualization_numview_{num_view}.png"

def analyze_temporal_consistency(num_view):
    """Analyze how num_view affects temporal consistency."""

    print(f"\n{'='*70}")
    print(f"TEMPORAL CONSISTENCY ANALYSIS: num_view={num_view}")
    print(f"{'='*70}")

    # Simulate attention patterns
    resolution = 384
    patch_size = 32
    patches_per_frame = (resolution // patch_size) ** 2

    print(f"\n📊 CONFIGURATION")
    print(f"  Resolution: {resolution}×{resolution} pixels")
    print(f"  Patch size: {patch_size}×{patch_size} pixels")
    print(f"  Patches per frame: {patches_per_frame}")
    print(f"  Frames in attention: {num_view}")

    print(f"\n🔍 ATTENTION MECHANICS")
    total_patches = patches_per_frame * num_view
    print(f"  Total patches: {total_patches}")
    print(f"  Attention matrix: {total_patches}×{total_patches}")
    print(f"  Total attention values: {total_patches**2:,}")

    # Calculate percentages
    within_frame = num_view * (patches_per_frame ** 2)
    cross_frame = (total_patches ** 2) - within_frame

    print(f"\n📈 ATTENTION DISTRIBUTION")
    print(f"  Within-frame attention: {within_frame:,} ({within_frame/(total_patches**2)*100:.1f}%)")
    print(f"  Cross-frame attention: {cross_frame:,} ({cross_frame/(total_patches**2)*100:.1f}%)")

    print(f"\n⚡ PERFORMANCE IMPACT")
    base_memory = (patches_per_frame ** 2) * 4  # bytes for float32
    total_memory = (total_patches ** 2) * 4
    print(f"  Memory per frame (no cross-frame): {base_memory / 1024:.1f} KB")
    print(f"  Memory with cross-frame: {total_memory / 1024:.1f} KB")
    print(f"  Memory multiplier: {total_memory / base_memory:.1f}×")

    # Temporal window analysis
    print(f"\n🎬 TEMPORAL WINDOW")
    print(f"  Each pixel can 'see': {num_view} frames")
    print(f"  Temporal context: ±{num_view//2} frames")
    print(f"  At 30 FPS: {num_view/30*1000:.1f}ms of video context")

def compare_num_view_quality():
    """Compare quality implications of different num_view settings."""

    print(f"\n{'='*70}")
    print("NUM_VIEW QUALITY COMPARISON")
    print(f"{'='*70}")

    comparisons = [
        {
            "num_view": 2,
            "pros": ["Fastest processing", "Lowest memory", "Good for static scenes"],
            "cons": ["Limited temporal context", "More flickering possible", "Less motion coherence"],
            "use_cases": ["Real-time applications", "Memory-constrained devices", "Static cameras"]
        },
        {
            "num_view": 3,
            "pros": ["Model default (trained with this)", "Balanced quality/speed", "Good temporal context"],
            "cons": ["Moderate memory usage", "Not optimal for fast motion"],
            "use_cases": ["General video processing", "Standard quality needs", "Most common scenarios"]
        },
        {
            "num_view": 4,
            "pros": ["Better motion handling", "Improved temporal consistency", "Smoother transitions"],
            "cons": ["Higher memory (1.8× of num_view=3)", "Slower processing", "May deviate from training"],
            "use_cases": ["High-quality output", "Fast motion scenes", "Professional production"]
        },
        {
            "num_view": 5,
            "pros": ["Maximum temporal context", "Best theoretical consistency"],
            "cons": ["Very high memory (2.8× of num_view=3)", "Much slower", "Untested territory"],
            "use_cases": ["Experimental", "Research purposes", "When quality is paramount"]
        }
    ]

    for config in comparisons:
        print(f"\n### num_view = {config['num_view']}")
        print(f"{'─'*50}")

        print("✅ PROS:")
        for pro in config['pros']:
            print(f"  • {pro}")

        print("❌ CONS:")
        for con in config['cons']:
            print(f"  • {con}")

        print("📋 USE CASES:")
        for use_case in config['use_cases']:
            print(f"  • {use_case}")

def explain_implementation():
    """Explain how num_view is implemented in the code."""

    print(f"\n{'='*70}")
    print("HOW NUM_VIEW IS IMPLEMENTED")
    print(f"{'='*70}")

    print("""
The num_view parameter controls cross-frame self-attention through tensor reshaping:

1️⃣ INPUT PROCESSING
   Original: (batch * num_view, seq_len, channels)
   Example:  (3, 1296, 768) for 3 frames at 384px

2️⃣ RESHAPE FOR ATTENTION
   ```python
   hidden_states = einops.rearrange(
       hidden_states,
       "(b n) hw c -> b (n hw) c",
       n=num_view
   )
   ```
   Result: (1, 3888, 768) - All frames concatenated

3️⃣ ATTENTION COMPUTATION
   Now each position can attend to ALL positions across ALL frames
   Q, K, V matrices: (1, 3888, 768)
   Attention: softmax(QK^T / sqrt(d)) * V

4️⃣ RESHAPE BACK
   ```python
   hidden_states = einops.rearrange(
       hidden_states,
       "b (n hw) c -> (b n) hw c",
       n=num_view
   )
   ```
   Result: (3, 1296, 768) - Back to per-frame format

🔑 KEY INSIGHT:
The reshape allows the standard attention mechanism to naturally
handle cross-frame relationships without modifying the core attention code.
""")

def generate_recommendation():
    """Generate recommendations for num_view usage."""

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR NUM_VIEW")
    print(f"{'='*70}")

    print("""
🎯 QUICK DECISION GUIDE:

1. DEFAULT CHOICE: num_view=3
   - This is what the model was trained with
   - Best balance for most use cases
   - Start here unless you have specific needs

2. WHEN TO USE num_view=2:
   - Processing on mobile/edge devices
   - Real-time requirements (<100ms per frame)
   - Static or slow-moving scenes
   - Memory < 4GB available

3. WHEN TO USE num_view=4:
   - Professional quality requirements
   - Fast camera or object motion
   - Have 8GB+ GPU memory
   - Can afford 2× processing time

4. EXPERIMENTAL IDEAS:

   a) Dynamic num_view based on motion:
      ```python
      motion_score = calculate_motion(frames)
      num_view = 2 if motion_score < 0.3 else 3 if motion_score < 0.7 else 4
      ```

   b) Pyramid-based num_view:
      - Use higher num_view for coarse levels
      - Lower num_view for fine details

   c) Adaptive based on GPU memory:
      ```python
      available_memory = torch.cuda.mem_get_info()[0]
      num_view = 4 if available_memory > 8e9 else 3 if available_memory > 4e9 else 2
      ```

⚠️ IMPORTANT NOTES:
- The model was trained with num_view=3
- Other values may reduce quality even if they seem theoretically better
- Always test with your specific content type
- Monitor memory usage closely with num_view > 3
""")

def main():
    """Run complete num_view analysis."""

    print("="*70)
    print("COMPLETE NUM_VIEW PARAMETER ANALYSIS")
    print("="*70)

    # Generate visualizations for different num_view values
    print("\n📊 Generating visualizations...")
    for nv in [2, 3, 4]:
        viz_path = visualize_attention_matrix(nv)
        print(f"  Created: {viz_path}")

    # Analyze temporal consistency
    for nv in [2, 3, 4]:
        analyze_temporal_consistency(nv)

    # Compare quality implications
    compare_num_view_quality()

    # Explain implementation
    explain_implementation()

    # Generate recommendations
    generate_recommendation()

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\nVisualizations saved to: output/num_view_analysis/")
    print("Use these insights to choose the optimal num_view for your use case.")

if __name__ == "__main__":
    main()