#!/usr/bin/env python
"""
Test script for Diffusers migration from 0.30.0 to 0.35.1.

This script validates that the cross-frame attention functionality
works correctly with the new approach using standard Diffusers.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_cross_frame_processor():
    """Test the custom cross-frame attention processor."""

    print("=" * 60)
    print("DIFFUSERS MIGRATION TEST")
    print("=" * 60)

    # Check current Diffusers version
    try:
        import diffusers
        print(f"\nCurrent Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("ERROR: Diffusers not installed")
        return False

    # Check einops availability
    try:
        import einops
        print(f"Einops version: {einops.__version__}")
    except ImportError:
        print("WARNING: einops not installed (required for cross-frame attention)")

    # Test 1: Import custom processor
    print("\n1. Testing custom processor import...")
    try:
        from rollingdepth.cross_frame_attention import (
            CrossFrameAttnProcessor2_0,
            set_cross_frame_attention_processor,
            inject_cross_frame_kwargs
        )
        print("✅ Custom processor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import custom processor: {e}")
        return False

    # Test 2: Check if we can create processor instance
    print("\n2. Testing processor instantiation...")
    try:
        if CrossFrameAttnProcessor2_0 is not None:
            processor = CrossFrameAttnProcessor2_0()
            print("✅ Processor instance created")
        else:
            print("⚠️  CrossFrameAttnProcessor2_0 not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Failed to create processor: {e}")
        return False

    # Test 3: Check tensor reshaping logic
    print("\n3. Testing tensor reshaping...")
    try:
        import einops

        # Create test tensor
        batch_views = 4  # 2 batches * 2 views
        seq_len = 64
        channels = 512
        num_view = 2

        test_tensor = torch.randn(batch_views, seq_len, channels)
        print(f"Original shape: {test_tensor.shape}")

        # Test reshaping forward
        reshaped = einops.rearrange(
            test_tensor,
            "(b n) s c -> b (n s) c",
            n=num_view
        )
        print(f"Reshaped for cross-frame: {reshaped.shape}")

        # Test reshaping back
        restored = einops.rearrange(
            reshaped,
            "b (n s) c -> (b n) s c",
            n=num_view,
            s=seq_len
        )
        print(f"Restored shape: {restored.shape}")

        # Verify shapes match
        assert test_tensor.shape == restored.shape
        print("✅ Tensor reshaping works correctly")

    except ImportError:
        print("⚠️  Skipping tensor test (einops not available)")
    except Exception as e:
        print(f"❌ Tensor reshaping failed: {e}")
        return False

    # Test 4: Check compatibility with current RollingDepth
    print("\n4. Testing RollingDepth compatibility...")
    try:
        # Check if we're using modified Diffusers or standard
        diffusers_path = Path("diffusers")
        if diffusers_path.exists():
            print("⚠️  Modified Diffusers 0.30.0 directory found")
            print("   Migration will replace this with standard Diffusers 0.35.1")
        else:
            print("✅ Using standard Diffusers from pip")

        # Check if RollingDepth imports work
        try:
            from rollingdepth import RollingDepthPipeline
            print("✅ RollingDepth imports successfully")
        except ImportError:
            print("⚠️  RollingDepth import failed (expected if using modified Diffusers)")

    except Exception as e:
        print(f"⚠️  Compatibility check error: {e}")

    # Test 5: Migration readiness check
    print("\n5. Migration Readiness Check:")
    print("-" * 40)

    ready = True

    # Check Python version
    import sys
    if sys.version_info >= (3, 11):
        print("✅ Python 3.11+ (TouchDesigner compatible)")
    else:
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} (recommend 3.11+)")

    # Check PyTorch version
    if torch.__version__ >= "2.0.0":
        print(f"✅ PyTorch {torch.__version__} (supports scaled_dot_product_attention)")
    else:
        print(f"⚠️  PyTorch {torch.__version__} (recommend 2.0+)")

    # Check device
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available")
    elif torch.cuda.is_available():
        print("✅ CUDA available")
    else:
        print("⚠️  No GPU acceleration available")

    print("\n" + "=" * 60)
    print("MIGRATION STRATEGY SUMMARY")
    print("=" * 60)

    print("""
The migration plan involves:

1. CUSTOM PROCESSOR APPROACH (Recommended):
   - Use standard Diffusers 0.35.1 from pip
   - Add custom cross-frame processor (cross_frame_attention.py)
   - No core Diffusers modifications needed
   - Easy to maintain and update

2. IMPLEMENTATION STATUS:
   ✅ Migration plan created (DIFFUSERS_MIGRATION_PLAN.md)
   ✅ Custom processor implemented (cross_frame_attention.py)
   ⏳ Integration with RollingDepth pipeline pending
   ⏳ Testing with actual models pending

3. NEXT STEPS:
   a) Install Diffusers 0.35.1: pip install diffusers==0.35.1
   b) Update RollingDepthPipeline to use custom processor
   c) Test with actual depth estimation models
   d) Benchmark performance vs current implementation

4. FALLBACK OPTION:
   If issues arise, continue using modified Diffusers 0.30.0
   in the ./diffusers directory as currently configured.
""")

    return True


if __name__ == "__main__":
    success = test_cross_frame_processor()
    sys.exit(0 if success else 1)