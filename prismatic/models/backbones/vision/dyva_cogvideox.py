"""
dyva_cogvideox.py

VisionBackbone based on CogVideoX-1.5-5B-I2V for PrismaticVLM.
This backbone runs the DiT only up to the middle layer and uses the output features.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Union, Dict, Optional, List
import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.distributed.fsdp.wrap import _module_wrap_policy

# Diffusers imports
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.utils import load_image

# Reuse utilities from your existing codebase
from prismatic.util.svd_utils import randn_tensor, SVDImageTransform
from prismatic.models.backbones.vision.base_vision import VisionBackbone
from prismatic.overwatch import initialize_overwatch

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import SSLError, ConnectionError
from urllib3.exceptions import MaxRetryError, ReadTimeoutError

NETWORK_EXCEPTIONS = (
    SSLError,
    ConnectionError,
    MaxRetryError,
    ReadTimeoutError,
    TimeoutError
)

overwatch = initialize_overwatch(__name__)

class EarlyExitException(Exception):
    """Custom exception to stop model execution early."""
    def __init__(self, features):
        self.features = features

class CogVideoXVisionBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        # CogVideoX specific params
        model_path: str = "THUDM/CogVideoX1.5-5B-I2V",
        torch_dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 32,
        height: int = 224,
        width: int = 224,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)

        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.height = height
        self.width = width
        self.num_frames = num_frames

        # 1. Load CogVideoX Pipeline components
        self.pipeline, self.vae, self.transformer = self._load_models()

        # 2. Set image transform
        self.image_transform = self._create_image_transform()

        # 3. Register Hook for Early Exit (Running only half the DiT)
        self._register_early_exit_hook()

        # 4. Move to device
        self.to(self.torch_dtype)
        # VAE and Text Encoder are frozen and can be kept in eval mode
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Cache empty prompt embeddings
        self.empty_prompt_embeds = None
    @retry(
        retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )

    def _load_models(self):
        """Load the CogVideoX pipeline components."""
        overwatch.info(f"Loading CogVideoX model from {self.model_path}...")
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype
        )
        return pipeline, pipeline.vae, pipeline.transformer

    def _register_early_exit_hook(self):
        """
        Register a forward hook on the middle block.
        """
        num_blocks = len(self.transformer.transformer_blocks)
        self.exit_idx = num_blocks // 2
        overwatch.info(f"CogVideoX DiT has {num_blocks} blocks. Will early exit after block {self.exit_idx}.")

        def hook_fn(module, input, output):
            raise EarlyExitException(output)

        self.transformer.transformer_blocks[self.exit_idx].register_forward_hook(hook_fn)

    def _create_image_transform(self) -> Callable:
        if self.image_resize_strategy == "resize-naive":
            single_frame_transform = Compose([
                Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            return SVDImageTransform(single_frame_transform)
        else:
            raise ValueError(f"Strategy {self.image_resize_strategy} not supported.")

    @torch.no_grad()
    def forward(self, input_pixel_values: torch.Tensor, prompt: str = "") -> torch.Tensor:
        """
        Use the CogVideoX official pipeline's prepare_latents to construct DiT inputs,
        then early-exit at the middle block to extract features.
        """
        device = input_pixel_values.device
        dtype = self.torch_dtype
        self.pipeline.to(device)

        # 1. Unify input handling for 4D (single image) and 5D (video frames) inputs
        if input_pixel_values.ndim == 5:
            # Input: [Batch, Frames, Channels, Height, Width]
            batch_size, num_input_frames, _, _, _ = input_pixel_values.shape
            first_frame = input_pixel_values[:, 0]
            if num_input_frames > 8:
                num_input_frames = 8
            # Prepare remaining frames for batch encoding (frames 1 to N)
            remaining_frames = input_pixel_values[:, 1:num_input_frames] if num_input_frames > 1 else None
        else:
            # Input: [Batch, Channels, Height, Width]
            batch_size, _, _, _ = input_pixel_values.shape
            first_frame = input_pixel_values
            remaining_frames = None
            num_input_frames = 1

        # 2. Initialize latents and background condition using the first frame
        # This sets up the base tensors (latents and image_latents) for the full video duration
        latents, image_latents = self.pipeline.prepare_latents(
            image=first_frame.to(device=device, dtype=dtype),
            batch_size=first_frame.shape[0],
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            dtype=dtype,
            device=device,
        )
        # shape: [B F_T=floor[(num_frames-1)/4+1] C=16 H//8 W//8]

        # 3. If multiple frames are provided, process remaining frames in a single batch (Speed Optimization)
        if remaining_frames is not None:
            # Flatten batch and frames dimensions: [B, F-1, C, H, W] -> [B*(F-1), C, H, W]
            flat_batch_size = remaining_frames.shape[0] * remaining_frames.shape[1]
            flat_frames = remaining_frames.flatten(0, 1).to(device=device, dtype=dtype)

            # Encode all remaining frames at once
            _, encoded_remaining = self.pipeline.prepare_latents(
                image=flat_frames,
                batch_size=flat_batch_size,
                num_frames=1,  # Treat each frame as an independent image
                height=self.height,
                width=self.width,
                dtype=dtype,
                device=device,
            )
            # Reshape encoded frames back to [B, F-1, C, H, W] for insertion
            encoded_remaining = encoded_remaining[:, 0, :, :, :]
            encoded_features = encoded_remaining.squeeze(1).reshape(batch_size, remaining_frames.shape[1], *encoded_remaining.shape[1:])

            # Calculate target indices in the latent temporal dimension
            total_latent_frames = image_latents.shape[1]
            # Generate indices for ALL frames, then slice [1:] for remaining frames
            all_indices = np.linspace(0, total_latent_frames - 1, num_input_frames, dtype=int)
            target_indices = all_indices[1:]

            # 4. Vectorized insertion: Replace latent slices all at once
            image_latents[:, target_indices, :, :, :] = encoded_features

        # 3. Apply scheduler.scale_model_input to be consistent with official behavior
        if hasattr(self.pipeline, "scheduler"):
            # The normal pipeline would call set_timesteps; here we default to 999 or you can call set_timesteps yourself
            t = torch.tensor([999], device=device, dtype=torch.long)
            t = t.expand(batch_size)
            # scheduler.scale_model_input expects shape [B, F, C, H, W]
            latents_scaled = self.pipeline.scheduler.scale_model_input(latents, t)
        else:
            t = torch.tensor([999] * batch_size, device=device, dtype=torch.long)
            latents_scaled = latents

        # 4. Concatenate into 32-channel hidden_states, note dimension order [B, F, C, H, W]
        hidden_states = torch.cat([latents_scaled, image_latents], dim=2)  # dim=2 is the channel dimension
        # hidden_states shape: [B, F_lat, 32, H_lat, W_lat]

        # 5. Text encoder_hidden_states: use an empty prompt
        if self.empty_prompt_embeds is None or self.empty_prompt_embeds.device != device:
            self.empty_prompt_embeds = self._encode_empty_prompt(prompt, batch_size, device, dtype)

        encoder_hidden_states = self.empty_prompt_embeds
        if encoder_hidden_states.shape[0] != batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)

        # 6. ofs (officially a long tensor of zeros)
        ofs = torch.zeros(batch_size, dtype=torch.long, device=device)

        # 7. image_rotary_emb ignored for now, pass None (also optional in official code)
        image_rotary_emb = None

        # 8. Call the transformer and rely on the hook for early exit at the middle block
        extracted_features = None
        try:
            self.transformer(
                hidden_states=hidden_states,            # [B, F, 32, H_lat, W_lat]
                encoder_hidden_states=encoder_hidden_states,
                timestep=t,                             # [B]
                ofs=ofs,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )
        except EarlyExitException as e:
            extracted_features = e.features

        if extracted_features is None:
            raise RuntimeError("Transformer finished without triggering EarlyExitException.")

        if isinstance(extracted_features, tuple):
            extracted_features = extracted_features[0]
        return extracted_features

    def _encode_empty_prompt(self, prompt, batch_size, device, dtype):
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=226,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.pipeline.text_encoder(text_input_ids)[0]
        return prompt_embeds.to(dtype=dtype, device=device)

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(_module_wrap_policy, module_classes={CogVideoXVisionBackbone})

    @property
    def embed_dim(self) -> int:
        return self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

    @property
    def num_patches(self) -> int:
        compression_t = self.vae.config.temporal_compression_ratio
        compression_s = 8
        patch_size = self.transformer.config.patch_size
        patch_size_t = self.transformer.config.patch_size_t

        h_lat = self.height // compression_s
        w_lat = self.width // compression_s
        f_lat = (self.num_frames - 1) // compression_t + 1
        f_lat_padded = math.ceil(f_lat / patch_size_t) * patch_size_t

        grid_h = h_lat // patch_size
        grid_w = w_lat // patch_size
        grid_t = f_lat_padded // patch_size_t

        return grid_h * grid_w * grid_t

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.height, self.width)

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.torch_dtype

# ==========================================
# Main Test Function
# ==========================================
if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    def main():
        print("=== Running Fixed Test ===")
        HEIGHT, WIDTH = 224, 224
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"Init Model (Dtype: {dtype})...")
        backbone = CogVideoXVisionBackbone(
            vision_backbone_id="test",
            image_resize_strategy="resize-naive",
            height=HEIGHT, width=WIDTH,
            torch_dtype=dtype
        ).to(device).eval()

        # Generate Random Image
        print("Generating Input...")
        img = Image.fromarray(np.random.randint(0,255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        img = [img]

        # Transform
        pixel_values = backbone.image_transform(img)
        if isinstance(pixel_values, dict): pixel_values = pixel_values['pixel_values']
        pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=dtype)

        print(f"Input Shape: {pixel_values.shape}")

        # Forward
        try:
            print("Calling Forward...")
            features = backbone(pixel_values)
            print(f"Success! Output Shape: {features.shape}")
            print(f"Expected Patches: {backbone.num_patches}")
            print(f"Embed Dim: {backbone.embed_dim}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    main()