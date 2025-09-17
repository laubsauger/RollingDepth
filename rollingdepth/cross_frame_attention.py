"""
Cross-frame attention processor for RollingDepth.

This module provides custom attention processors that enable cross-frame/multi-view
attention in Diffusers 0.35.1, preserving the functionality from the modified
Diffusers 0.30.0 while using the standard Diffusers library.
"""

import torch
from typing import Optional, Dict, Any

# Check if einops is available
try:
    import einops
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("Warning: einops not available. Cross-frame attention will be disabled.")


def create_cross_frame_processor(base_processor_class):
    """
    Factory function to create a cross-frame enabled processor from any base class.

    This allows us to extend any existing attention processor (AttnProcessor2_0,
    XFormersAttnProcessor, etc.) with cross-frame capabilities.
    """

    class CrossFrameProcessor(base_processor_class):
        """
        Custom attention processor that adds cross-frame/multi-view support
        to any base attention processor class.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cross_frame_enabled = EINOPS_AVAILABLE

        def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            num_view: Optional[int] = None,
            *args,
            **kwargs
        ) -> torch.FloatTensor:
            """
            Process attention with optional cross-frame support.

            Args:
                attn: The attention module
                hidden_states: Input hidden states
                encoder_hidden_states: Encoder hidden states for cross-attention
                attention_mask: Attention mask
                temb: Time embedding
                num_view: Number of views/frames for cross-frame attention
                *args, **kwargs: Additional arguments for the base processor

            Returns:
                Processed hidden states
            """

            # Check if cross-frame should be applied
            if self.cross_frame_enabled and num_view is not None and num_view > 1:
                # Always reshape hidden_states for cross-frame (matching original)
                hidden_states = einops.rearrange(hidden_states, "(b n) hw c -> b (n hw) c", n=num_view)

                # Process with parent class
                hidden_states = super().__call__(
                    attn=attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    *args,
                    **kwargs
                )

                # Always reshape back (matching original)
                hidden_states = einops.rearrange(hidden_states, "b (n hw) c -> (b n) hw c", n=num_view)
            else:
                # Standard processing without cross-frame
                hidden_states = super().__call__(
                    attn=attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    *args,
                    **kwargs
                )

            return hidden_states

    # Set a meaningful name for the class
    CrossFrameProcessor.__name__ = f"CrossFrame{base_processor_class.__name__}"

    return CrossFrameProcessor


# Try to import and create cross-frame versions of common processors
try:
    from diffusers.models.attention_processor import (
        AttnProcessor2_0,
        AttnProcessor,
    )

    # Create cross-frame enabled versions
    CrossFrameAttnProcessor2_0 = create_cross_frame_processor(AttnProcessor2_0)
    CrossFrameAttnProcessor = create_cross_frame_processor(AttnProcessor)

except ImportError as e:
    print(f"Warning: Could not import attention processors from diffusers: {e}")
    CrossFrameAttnProcessor2_0 = None
    CrossFrameAttnProcessor = None

# Try to import XFormers processor if available
try:
    from diffusers.models.attention_processor import XFormersAttnProcessor
    CrossFrameXFormersAttnProcessor = create_cross_frame_processor(XFormersAttnProcessor)
except ImportError:
    CrossFrameXFormersAttnProcessor = None


def set_cross_frame_attention_processor(
    model,
    num_view: int = 2,
    processor_type: str = "auto"
) -> None:
    """
    Set cross-frame attention processors on a model (typically UNet).

    Args:
        model: The model to set processors on (e.g., UNet)
        num_view: Number of views/frames for cross-frame attention
        processor_type: Type of processor to use ("auto", "2.0", "xformers", "standard")
    """

    if not EINOPS_AVAILABLE:
        print("Warning: einops not installed. Cross-frame attention disabled.")
        return

    # Determine which processor to use
    if processor_type == "auto":
        # Auto-detect best processor
        if CrossFrameXFormersAttnProcessor is not None and model.device.type == "cuda":
            processor_class = CrossFrameXFormersAttnProcessor
            print("Using CrossFrameXFormersAttnProcessor")
        elif CrossFrameAttnProcessor2_0 is not None and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            processor_class = CrossFrameAttnProcessor2_0
            print("Using CrossFrameAttnProcessor2_0")
        else:
            processor_class = CrossFrameAttnProcessor
            print("Using CrossFrameAttnProcessor")
    elif processor_type == "xformers" and CrossFrameXFormersAttnProcessor is not None:
        processor_class = CrossFrameXFormersAttnProcessor
    elif processor_type == "2.0" and CrossFrameAttnProcessor2_0 is not None:
        processor_class = CrossFrameAttnProcessor2_0
    else:
        processor_class = CrossFrameAttnProcessor

    if processor_class is None:
        print("Error: No suitable attention processor available")
        return

    # Create processor instance
    processor = processor_class()

    # Set processor for all attention layers
    model.set_attn_processor(processor)

    print(f"Set cross-frame attention with num_view={num_view}")


def inject_cross_frame_kwargs(kwargs: Dict[str, Any], num_view: int) -> Dict[str, Any]:
    """
    Helper function to inject num_view into cross_attention_kwargs.

    Args:
        kwargs: Original kwargs dict
        num_view: Number of views/frames

    Returns:
        Modified kwargs with num_view injected
    """
    cross_attention_kwargs = kwargs.get('cross_attention_kwargs', {})
    cross_attention_kwargs['num_view'] = num_view
    kwargs['cross_attention_kwargs'] = cross_attention_kwargs
    return kwargs