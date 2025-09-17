# Copyright 2024 Dominik Narnhofer, Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2025-03-07
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/RollingDepth#-citation
# More information about the method can be found at https://rollingdepth.github.io
# ---------------------------------------------------------------------------------


import torch
from torch.optim.adam import Adam
import logging
import time
from typing import List, Tuple
from tqdm import tqdm


class DepthAligner:
    def __init__(
        self,
        device: torch.device,
        factor: int = 10,
        lmda: float = 1e-1,
        lmda2: float = 1e-1,
        lmda3: float = 1e1,
        lr: float = 1e-3,
        num_iterations: int = 2000,
        border: int = 2,
        verbose: bool = False,
        depth_loss_weight: float = 1.0,
        loss_scale=1.0,
        quality_mode: str = "balanced",
    ):
        self.factor = factor  # Depth down-scale factor for s,t computation
        self.lmda = lmda  # Controls soft constraints on s and t
        self.lr = lr  # Optimizer step size
        self.num_iterations = num_iterations  # Optimizer iterations
        self.border = border  # Num pixels for border crop
        self.verbose = verbose
        self.device = device
        self.lmda2 = lmda2
        self.depth_loss_weight = depth_loss_weight
        self.loss_scale = loss_scale
        self.lmda3 = lmda3
        self.quality_mode = quality_mode

    # Create indices to keep data structures simple
    def create_triplet_indices(self, sequence_length: int, gap: int, window_size: int):
        gap += 1  # Adjust gap for inclusive indexing
        index_list = []

        for i in range(sequence_length - (window_size - 1) * gap):
            indices = [i + j * gap for j in range(window_size)]
            index_list.append(indices)

        indices = torch.tensor(index_list)
        return indices

    def run(self, snippet_ls: List[torch.Tensor], dilations: List[int], seq_len: int = None, snippet_indices: List[List[List[int]]] = None):
        device = self.device

        # MPS optimization: Log timing if verbose
        if self.verbose and device.type == "mps":
            t_start = time.time()

        snippet_lenghts = [snippet.shape[1] for snippet in snippet_ls]
        gaps = [d - 1 for d in dilations]

        # If seq_len is provided, use it (correct for overlapping snippets)
        # Otherwise, calculate based on snippet data
        if seq_len is not None:
            sequence_length = seq_len
            if self.verbose:
                print(f"Using provided sequence_length = {sequence_length}")
        else:
            # Fallback calculation for non-overlapping snippets
            num_snippets = len(snippet_ls[0])
            snippet_len = snippet_lenghts[0]
            sequence_length = num_snippets * snippet_len - num_snippets + 1
            if self.verbose:
                print(f"WARNING: seq_len not provided, using fallback calculation = {sequence_length}")

        mn = min([snippet.min() for snippet in snippet_ls])  # type: ignore
        snippet_ls = [tmp - mn for tmp in snippet_ls]

        # Exclude border artifact
        triplets_scaled = [
            tmp[:, :, :, self.border : -self.border, self.border : -self.border].to(
                device
            )
            for tmp in snippet_ls
        ]

        # Scale down to factor size
        triplets_scaled = [
            tmp[:, :, :, :: self.factor, :: self.factor] for tmp in triplets_scaled
        ]

        # Create or use provided triplet indices
        if snippet_indices is not None:
            # Use the actual snippet indices from the pipeline
            indices_list = []
            for dilation_indices in snippet_indices:
                # Convert list of lists to tensor
                indices_tensor = torch.tensor(dilation_indices, dtype=torch.long)
                indices_list.append(indices_tensor)
        else:
            # Fallback to old method (may not work correctly with overlapping snippets)
            indices_list = [
                self.create_triplet_indices(sequence_length, g, w)
                for g, w in zip(gaps, snippet_lenghts)
            ]

        scales, translations, loss_history = self.optimize(
            snippet_ls=triplets_scaled,
            indices_list=indices_list,
            sequence_length=sequence_length,
        )

        merged_scaled_triplets = self.merge_scaled_triplets(
            snippet_ls=snippet_ls,
            indices_list=indices_list,
            s_list=scales,
            t_list=translations,
            sequence_length=sequence_length,
            device=torch.device("cpu"),
        )

        if self.verbose and device.type == "mps":
            t_end = time.time()
            logging.info(f"Co-alignment optimization took {t_end - t_start:.2f}s on MPS")

        if self.verbose:
            print(f"DEBUG: merged_scaled_triplets shape: {merged_scaled_triplets.shape}")
            print(f"DEBUG: Expected {sequence_length} frames")

        # Ensure we return the correct number of frames
        if merged_scaled_triplets.shape[0] != sequence_length:
            if self.verbose:
                print(f"WARNING: Expected {sequence_length} frames but got {merged_scaled_triplets.shape[0]}")

        return (
            merged_scaled_triplets,
            scales,
            translations,
            loss_history,
        )

    # Scaling Optimizer
    def optimize(
        self,
        snippet_ls: List[torch.Tensor],
        indices_list: List[torch.Tensor],
        sequence_length: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple]]:
        device = self.device
        H, W = snippet_ls[0].shape[-2], snippet_ls[0].shape[-1]
        windows = [triplet.shape[1] for triplet in snippet_ls]

        snippet_ls = [
            triplet.reshape(triplet.shape[0], w, H * W)
            for triplet, w in zip(snippet_ls, windows)
        ]
        scales = [
            torch.ones(A.shape[0], 1, 1, device=device, requires_grad=True)
            for A in snippet_ls
        ]
        translations = [
            torch.zeros(A.shape[0], 1, 1, device=device, requires_grad=True)
            for A in snippet_ls
        ]

        optimizer = Adam(scales + translations, lr=self.lr, betas=(0.5, 0.9))

        loss_ls = []

        def closure():
            # MPS optimization: Profile timing on first iteration
            if device.type == "mps" and len(loss_ls) == 0 and self.verbose:
                t_closure_start = time.time()

            A_scaled = [
                reshaped_tensor * s + t
                for reshaped_tensor, s, t in zip(snippet_ls, scales, translations)
            ]

            # MPS optimization: Pre-allocate tensors once
            total_w = sum(windows)
            M = torch.zeros(total_w, sequence_length, H * W, device=device)
            M_depth = torch.zeros(total_w, sequence_length, H * W, device=device)
            B = torch.zeros(total_w, sequence_length, H * W, device=device)

            for i, (scaled_tensor, indices, w) in enumerate(
                zip(A_scaled, indices_list, windows)
            ):
                M_depth[torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T] = (
                    scaled_tensor.clip(1e-3).permute(1, 0, 2)
                ) ** -1
                M[
                    torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T
                ] = scaled_tensor.permute(1, 0, 2)
                B[torch.arange(i * w, (i + 1) * w)[:, None], indices.long().T] = 1

            summ = M.sum(0) / B.sum(0)
            summ_depth = M_depth.sum(0) / B.sum(0)

            # Calculate target
            with torch.no_grad():
                # MPS optimization: Avoid unnecessary clone() operations
                target = summ.detach()
                target_depth = summ_depth.detach()
                scale = target.abs().mean(-1, keepdim=True)
                scale_depth = target_depth.abs().mean(-1, keepdim=True)

            loss = torch.abs((M - target) * B / scale).mean()
            loss_depth = torch.abs((M_depth - target_depth) * B / scale_depth).mean()

            loss = (loss + self.depth_loss_weight * loss_depth).mean()

            soft_constraints = sum(
                self.lmda2 * (torch.max(torch.tensor(0.0), 1 - s) ** 2).mean()
                + self.lmda3 * (t**2).mean()
                for s, t in zip(scales, translations)
            )

            loss = self.loss_scale * loss + soft_constraints
            loss.backward()
            loss_ls.append((loss.item(), summ.min().item(), summ.max().item()))
            return loss

        # Optimization loop
        iterable = range(self.num_iterations)
        if self.verbose:
            iterable = tqdm(iterable, desc="Co-align snippets", leave=False)

        # MPS optimization: Use more efficient gradient clearing and early stopping
        prev_loss = float('inf')
        patience_counter = 0

        # Configurable early stopping based on quality preference
        quality_mode = self.quality_mode

        # Set defaults
        max_iterations = self.num_iterations

        if quality_mode == 'fast':
            # Aggressive early stopping (~500-600 iterations)
            patience = 30 if device.type == "mps" else 50
            min_iterations = 500 if device.type == "mps" else 1000
            loss_threshold = 1e-4 if device.type == "mps" else 1e-5
            max_iterations = 700 if device.type == "mps" else 1000
        elif quality_mode == 'balanced':
            # Middle ground - force stop at 1000-1200 iterations
            patience = 50 if device.type == "mps" else 100
            min_iterations = 800 if device.type == "mps" else 1200
            loss_threshold = 5e-5 if device.type == "mps" else 1e-5
            max_iterations = 1200 if device.type == "mps" else 1500
        else:  # 'quality' or default
            # High quality - full iterations
            patience = 200 if device.type == "mps" else 300  # Very high patience
            min_iterations = 1500 if device.type == "mps" else 1800
            loss_threshold = 1e-6 if device.type == "mps" else 1e-7  # Very strict
            max_iterations = self.num_iterations  # Use full 2000

        if self.verbose and device.type == "mps":
            logging.info(f"Co-alignment quality mode: {quality_mode} (min_iter={min_iterations}, max_iter={max_iterations}, patience={patience})")

        for i in iterable:
            # Check if we've reached max iterations for this quality mode
            if i >= max_iterations:
                if self.verbose:
                    logging.info(f"Reached max iterations ({max_iterations}) for {quality_mode} mode")
                break

            if device.type == "mps":
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            else:
                optimizer.zero_grad()
            optimizer.step(closure)

            # Sync every 100 iterations on MPS to free memory
            if i % 100 == 0 and device.type == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            # Early stopping - check convergence
            if i > min_iterations and len(loss_ls) > 0:
                current_loss = loss_ls[-1][0]
                if abs(current_loss - prev_loss) < loss_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            logging.info(f"Early stopping at iteration {i} (converged)")
                        break
                else:
                    patience_counter = 0
                prev_loss = current_loss

            if i % 10 == 0 and self.verbose:
                logging.debug(
                    f"Iteration {i}, Loss_diff: {loss_ls[-1][0]:.6f}, Min: {loss_ls[-1][1]:.6f}, Max: {loss_ls[-1][2]:.6f}"
                )

        return scales, translations, loss_ls

    def merge_scaled_triplets(
        self,
        snippet_ls: List[torch.Tensor],
        indices_list: List[torch.Tensor],
        s_list: List[torch.Tensor],
        t_list: List[torch.Tensor],
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Merge overlapping snippet predictions.

        The key insight: with stride=1 and snippet_len=3, for 5 frames we get:
        - Snippet 0: frames [0,1,2]
        - Snippet 1: frames [1,2,3]
        - Snippet 2: frames [2,3,4]

        Each snippet in snippet_ls[i] has shape (num_snippets, snippet_len, C, H, W)
        indices_list[i] tells us which frames each snippet position corresponds to.
        """
        snippet_ls = [a.to(device) for a in snippet_ls]
        dtype = snippet_ls[0].dtype

        scales = s_list
        translations = t_list

        # Apply scale and translation
        A_scaled = [
            (
                reshaped_tensor * s[:, None, None].to(dtype).to(device)
                + t[:, None, None].to(dtype).to(device)
            )
            for reshaped_tensor, s, t in zip(snippet_ls, scales, translations)
        ]

        # Get dimensions
        C = A_scaled[0].shape[2] if len(A_scaled[0].shape) > 2 else 1
        H = A_scaled[0].shape[3] if len(A_scaled[0].shape) > 3 else 1
        W = A_scaled[0].shape[4] if len(A_scaled[0].shape) > 4 else 1

        # Initialize accumulator for averaging overlapping predictions
        frame_sum = {}  # frame_idx -> list of predictions

        for i_dilation in range(len(A_scaled)):
            # Each dilation level has its own set of snippets
            snippets = A_scaled[i_dilation]  # shape: (num_snippets, snippet_len, C, H, W)
            indices = indices_list[i_dilation]  # shape: (num_snippets, snippet_len)

            for snippet_idx in range(len(indices)):
                for pos_in_snippet in range(len(indices[snippet_idx])):
                    frame_idx = int(indices[snippet_idx, pos_in_snippet].item())

                    if 0 <= frame_idx < sequence_length:
                        if frame_idx not in frame_sum:
                            frame_sum[frame_idx] = []
                        frame_sum[frame_idx].append(snippets[snippet_idx, pos_in_snippet])

        # Average overlapping predictions and build sequence
        seq = []
        for i_frame in range(sequence_length):
            if i_frame in frame_sum:
                # Average all predictions for this frame
                frame_preds = torch.stack(frame_sum[i_frame])
                avg_pred = frame_preds.mean(0)
                seq.append(avg_pred)
            else:
                # This shouldn't happen if indices are correct
                if self.verbose:
                    print(f"Warning: Frame {i_frame} not found in any snippet")
                # Create a zero frame as fallback
                seq.append(torch.zeros((C, H, W), device=device, dtype=dtype))

        # Stack and reshape to match expected format
        # The pipeline expects shape (N, 1, H, W) where N is number of frames
        if seq:
            # Stack all frames
            result = torch.stack(seq)  # (seq_len, C, H, W) or (seq_len, H, W)

            # Handle different tensor shapes
            if len(result.shape) == 3:  # (seq_len, H, W)
                # Add channel dimension
                result = result.unsqueeze(1)  # (seq_len, 1, H, W)
            elif len(result.shape) == 4 and result.shape[1] > 1:  # (seq_len, C>1, H, W)
                # If there are multiple channels, we need to handle this differently
                # For depth, we typically expect single channel
                result = result.mean(dim=1, keepdim=True)  # Average channels to get (seq_len, 1, H, W)

            # Result should now be (seq_len, 1, H, W)
            return result
        else:
            # Return empty tensor in the expected format
            return torch.zeros((sequence_length, 1, H, W), device=device, dtype=dtype)
