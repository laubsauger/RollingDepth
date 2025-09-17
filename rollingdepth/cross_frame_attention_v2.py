"""
Cross-frame attention for Diffusers 0.35.1 - Alternative approach.

Instead of modifying the processor, we patch the Attention forward method
to handle cross-frame reshaping at the right level.
"""

import torch
from typing import Optional, Dict, Any
import einops


def patch_attention_for_cross_frame():
    """
    Patch the Attention class to support cross-frame attention.
    This approach modifies the Attention.forward method to handle num_view.
    """
    from diffusers.models.attention import Attention

    # Store original forward method
    original_forward = Attention.forward

    def cross_frame_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modified forward that handles cross-frame attention.
        """
        # Extract num_view from cross_attention_kwargs
        num_view = None
        if cross_attention_kwargs:
            num_view = cross_attention_kwargs.get('num_view', None)

        # Store original shape for residual
        residual = hidden_states

        # For cross-frame attention, we need to reshape BEFORE processing
        # but preserve the residual shape
        if num_view is not None and num_view > 1:
            # Only reshape for self-attention
            is_self_attn = encoder_hidden_states is None

            if is_self_attn:
                # Reshape hidden states for cross-frame processing
                hidden_states = einops.rearrange(
                    hidden_states, "(b n) hw c -> b (n hw) c", n=num_view
                )

        # Call original forward
        hidden_states = original_forward(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs
        )

        # Reshape back after processing
        if num_view is not None and num_view > 1:
            is_self_attn = encoder_hidden_states is None
            if is_self_attn:
                hidden_states = einops.rearrange(
                    hidden_states, "b (n hw) c -> (b n) hw c", n=num_view
                )

        return hidden_states

    # Replace the forward method
    Attention.forward = cross_frame_forward
    print("Patched Attention.forward for cross-frame support")


def patch_basic_transformer_for_cross_frame():
    """
    Alternative: Patch BasicTransformerBlock to handle cross-frame at block level.
    """
    from diffusers.models.attention import BasicTransformerBlock

    # Store original forward
    original_forward = BasicTransformerBlock.forward

    def cross_frame_block_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        class_labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Modified BasicTransformerBlock forward that handles num_view properly.
        """
        # Extract num_view
        num_view = None
        if cross_attention_kwargs:
            num_view = cross_attention_kwargs.get('num_view', None)

        if num_view is not None and num_view > 1:
            # Store shape
            batch_views, seq_len, channels = hidden_states.shape

            # Only reshape for blocks that will do self-attention
            # Check if this is first attention (self-attention)
            if not self.only_cross_attention:
                # Reshape for cross-frame
                hidden_states = einops.rearrange(
                    hidden_states, "(b n) s c -> b (n s) c", n=num_view
                )

            # Process
            hidden_states = original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                **kwargs
            )

            # Reshape back
            if not self.only_cross_attention:
                hidden_states = einops.rearrange(
                    hidden_states, "b (n s) c -> (b n) s c",
                    n=num_view, s=seq_len
                )
        else:
            # Standard processing
            hidden_states = original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                **kwargs
            )

        return hidden_states

    # Replace forward method
    BasicTransformerBlock.forward = cross_frame_block_forward
    print("Patched BasicTransformerBlock.forward for cross-frame support")


def enable_cross_frame_attention():
    """
    Enable cross-frame attention by patching Diffusers classes.
    """
    try:
        import einops
        # Try the simpler Attention-level patch first
        patch_attention_for_cross_frame()
        return True
    except Exception as e:
        print(f"Failed to patch for cross-frame attention: {e}")
        return False