"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn
from typing import Dict # type: ignore

# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        print(f"vision_dim = {fused_vision_dim}, llm_dim = {llm_dim}, mlp_type = {mlp_type}")
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)
    
class DualProjector(nn.Module):
    def __init__(self, fused_vision_dim: Dict[str, int], llm_dim: int):
        super().__init__()
        
        svd_initial_dim = fused_vision_dim["svd"] * 4
        self.svd_projector_base = nn.Sequential(
            nn.Linear(fused_vision_dim["svd"], svd_initial_dim),
            nn.GELU(),
            nn.Linear(svd_initial_dim, llm_dim),
            nn.GELU(),
        )
        
        other_key = next(key for key in fused_vision_dim if key != 'svd')
        self.other_key = other_key
        
        other_initial_dim = fused_vision_dim[other_key] * 4
        self.projector_base = nn.Sequential(
            nn.Linear(fused_vision_dim[other_key], other_initial_dim),
            nn.GELU(),
            nn.Linear(other_initial_dim, llm_dim),
            nn.GELU(),
        )
        
        self.final_projection = nn.Linear(llm_dim, llm_dim)

    @torch._dynamo.disable
    def forward(self, fused_img_patches: Dict[str, torch.Tensor]) -> torch.Tensor:
        svd_features = self.svd_projector_base(fused_img_patches["svd"])
        other_features = self.projector_base(fused_img_patches[self.other_key])
        fused_features = torch.cat([other_features, svd_features], dim=1)
        output = self.final_projection(fused_features)
        
        return output

class BatchAwareGuidanceFusion(nn.Module):
    """
    Handles batched classifier-free guidance fusion.
    Assumes input is a tensor where the first half is the unconditional batch
    and the second half is the conditional batch.
    """
    def __init__(self):
        super().__init__()
        # Ensure guidance_scale is a float for the formula
        self.guidance_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies classifier-free guidance.
        
        Input shape: (2*B, seq_len, C) 
        - Where first B elements are unconditional samples
        - Next B elements are conditional samples
        
        Output shape: (B, seq_len, C) after fusion
        """
        # Ensure input has correct dimensions
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (2*B, seq_len, C), got {x.dim()}D tensor")
            
        # Split batch into unconditional and conditional halves
        batch_size = x.shape[0]
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for guidance fusion, got {batch_size}")
            
        unconditional, conditional = x.chunk(2, dim=0)
        
        # Verify the split worked correctly
        assert conditional.shape == unconditional.shape, \
            f"Chunk sizes for guidance fusion mismatch: {conditional.shape} vs {unconditional.shape}"
            
        # Standard guidance formula: uncond + scale * (cond - uncond)
        # Note: The original formula was `cond + scale * (uncond - cond)`, which is unconventional.
        # The standard formula is used here for correctness. If the original was intended, swap the terms.
        fused = unconditional + self.guidance_scale * (conditional - unconditional)
        return fused # Shape: (B, seq_len, C)

class ReshapeToSequence(nn.Module):
    """
    Reshapes a 4D tensor from (B, C, H, W) to a 3D sequence tensor (B, H*W, C).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor")
        
        B, C, H, W = x.shape
        # Permute to bring channels to the last dimension, then reshape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

# --- Refactored FusedMLPProjector ---

class SVDProjector(nn.Module):
    """
    A comprehensive vision projector that combines a convolutional encoder,
    reshaping, MLP projection, and optional classifier-free guidance.
    
    Processes a 4D vision tensor into a 3D sequence suitable for a language model.
    """
    def __init__(self, vision_dim: int, llm_dim: int,) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        
        modules = []

        # Part 1: Convolutional Encoder to reduce spatial dimensions
        # Get configuration parameters from the config object or use defaults
        in_channels = self.vision_dim
        hidden_channels = 256
        out_channels = 512
        
        conv_layers = [
            # First conv: in_channels -> hidden_channels, stride 2 reduces spatial dim by half
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # Second conv: hidden_channels -> hidden_channels*2, stride 2
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # Third conv: hidden_channels*2 -> out_channels, stride 2
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        ]
        modules.extend(conv_layers)
        
        # Part 2: Reshape from 4D image grid to 3D sequence
        modules.append(ReshapeToSequence())

        # Part 3: MLP to project from vision feature dimension to LLM dimension
        # The input dimension for the MLP is the number of output channels from the conv block.
        mlp_input_dim = out_channels
        initial_projection_dim = mlp_input_dim * 4 # Using the 4x expansion factor from your original class

        mlp_projector = nn.Sequential(
            nn.Linear(mlp_input_dim, initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(initial_projection_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )
        modules.append(mlp_projector)

        # Part 4: Add classifier-free guidance if specified in the config
        modules.append(BatchAwareGuidanceFusion())

        # Combine all parts into a single sequential model
        self.projector = nn.Sequential(*modules)
        
        print(f"Initialized FusedMLPProjector with ConvEncoder -> Reshape -> MLP -> GuidanceFusion")
        print(self.projector)


    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the vision projector.
        
        Args:
            image_features (torch.Tensor): Input tensor from vision encoder.
                                            Shape: (B, C_in, H_in, W_in)
        
        Returns:
            torch.Tensor: Processed tensor ready for the language model.
                        Shape (with guidance): (B/2, Seq_len_out, llm_dim)
                        Shape (without guidance): (B, Seq_len_out, llm_dim)
        """
        return self.projector(image_features)