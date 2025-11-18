"""
dyva.py

VisionBackbone based on SigLIP and Stable Video Diffusion (SVD) for PrismaticVLM.
This backbone fuses intermediate UNet features from SVD with SigLIP features as visual representations.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple, Optional, List, Union, Any

import timm
import torch
import torch.nn as nn
import PIL
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                        transformer_auto_wrap_policy)
from torchvision.transforms import Compose, Resize
from torchvision import transforms
from torchvision.transforms import functional as TF

from prismatic.models.backbones.vision.base_vision import (ImageTransform,
                                                            VisionBackbone,
                                                            unpack_tuple)
from prismatic.models.backbones.vision.dyva_svd import SVDVisionBackbone


import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import SSLError, ConnectionError
from urllib3.exceptions import MaxRetryError, ReadTimeoutError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
NETWORK_EXCEPTIONS = (
    SSLError,
    ConnectionError,
    MaxRetryError,
    ReadTimeoutError,
    TimeoutError
)


# Registry =>> Supported DyVA Pairs
# Defines model identifiers and the input image sizes for the SVD featurizer
SIGLIPSVD_VISION_BACKBONES = {
    "dyva_siglip": {
        "siglip": "vit_so400m_patch14_siglip_224",
        "svd_height": 448,
        "svd_width": 448,
    },
    "dyva_dino": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "svd_height": 448,
        "svd_width": 448,
    },
    "dyva_clip": {
        "clip": "vit_large_patch14_clip_224.openai",
        "svd_height": 448,
        "svd_width": 448,
    },
}

@dataclass
class DyVAImageTransform:
    transforms: Dict[str, Any]
    resize_strategy: str
    patch_size: int
    is_prismatic: bool = True

    def __call__(self, img: Union[PIL.Image.Image, List[PIL.Image.Image]], **kwargs: str) -> Dict[str, torch.Tensor]:
        img_for_vit = img[0] if isinstance(img, List) else img
        img_for_svd = img

        transformed_img = {}
        for name, transform in self.transforms.items():
            if name == "svd":
                if not isinstance(img_for_svd, list):
                    img_for_svd = [img_for_svd]
                ts_img = transform(img_for_svd, **kwargs)
                transformed_img[name] = ts_img
            else:
                transformed_img[name] = transform(img_for_vit, **kwargs)
        return transformed_img

SUPPORTED_VIT_MODELS = ["dino", "siglip", "clip"]

class DyVAVisionBackbone(VisionBackbone):
    """
    A Vision Backbone that combines SigLIP and SVD features.
    It processes images in parallel and returns concatenated features extracted from the two models.
    """
    def __init__(self,
                vision_backbone_id: str,
                image_resize_strategy: str,
                default_image_size: int = 224,
                num_frames: Optional[int] = 8,) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.vision_backbone_id = vision_backbone_id
        if self.vision_backbone_id not in SIGLIPSVD_VISION_BACKBONES:
            raise ValueError(f"Vision backbone ID `{self.vision_backbone_id}` is not defined in SIGLIPSVD_VISION_BACKBONES.")

        model_cfg = SIGLIPSVD_VISION_BACKBONES[self.vision_backbone_id]
        svd_height = model_cfg["svd_height"]
        svd_width = model_cfg["svd_width"]

        if self.image_resize_strategy not in ['resize-naive']:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported.")

        self.featurizers = nn.ModuleDict()
        image_transforms = {}

        self.vision_cnt = 0
        for model_type in SUPPORTED_VIT_MODELS:
            if model_type in model_cfg:
                self.vision_cnt += 1

        self.vit_component_keys: List[str] = []
        for model_type in SUPPORTED_VIT_MODELS:
            if model_type in model_cfg:
                featurizer, transform = self._create_vision_encoder(model_cfg, model_type)
                self.featurizers[model_type] = featurizer
                image_transforms[model_type] = transform
                self.vit_component_keys.append(model_type)
        if not self.vit_component_keys:
            raise ValueError(f"Configuration for `{self.vision_backbone_id}` must include at least one of {SUPPORTED_VIT_MODELS}.")

        # 2. Initialize the SVD Featurizer
        svd_featurizer = SVDVisionBackbone(
            vision_backbone_id="dyva-svd",
            image_resize_strategy="resize-naive",
            height=svd_height,
            width=svd_width,
            num_frames=num_frames,
        )
        svd_featurizer.eval()
        self.featurizers["svd"] = svd_featurizer
        image_transforms["svd"] = self.featurizers["svd"].get_image_transform()

        # 3. Create image transforms for both models
        self.image_transform = DyVAImageTransform(
            transforms=image_transforms,
            resize_strategy=self.image_resize_strategy,
            patch_size=self.default_image_size,
        )

        # 4. Store the first ViT model's data config for default resolution
        self.first_vit_key = self.vit_component_keys[0]
        self.first_vit_data_cfg = timm.data.resolve_model_data_config(self.featurizers[self.first_vit_key])
        self.first_vit_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

    @retry(
        retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            f"Failed to load model due to: {retry_state.outcome.exception()}. "
            f"Retrying attempt {retry_state.attempt_number} after {retry_state.next_action.sleep:.2f} seconds..."
        )
    )

    def _create_model_with_retry(self, url: str) -> VisionTransformer:
        """Create a model with retry using tenacity."""
        model: VisionTransformer = timm.create_model(
            url,
            pretrained=True,
            num_classes=0,
            img_size=self.default_image_size
        )
        return model

    def _create_vision_encoder(self, model_cfg: Dict, model_type: str) -> Tuple[VisionTransformer, ImageTransform]:
        """
        Create the vision backbone and image transform for the given model type (siglip/dino/clip).
        """
        logger.info(f"Creating Vision Encoder for `{model_type}`...")

        # Load model
        timm_path_or_url = model_cfg[model_type]
        featurizer = self._create_model_with_retry(timm_path_or_url)
        self.total_num_patches = featurizer.patch_embed.num_patches
        featurizer.eval()

        # Modify forward to return the penultimate layer features
        # Note: `get_intermediate_layers` for timm >= 0.10.0, n={-2} is a more modern usage
        featurizer.forward = unpack_tuple(
            partial(featurizer.get_intermediate_layers, n={len(featurizer.blocks) - 2})
        )

        # Load image transform
        model_data_cfg = timm.data.resolve_model_data_config(featurizer)
        model_data_cfg["input_size"] = (self.default_image_size, self.default_image_size)
        default_transform = timm.data.create_transform(**model_data_cfg, is_training=False)

        assert isinstance(default_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_transform.transforms[0], Resize)

        # Ensure input images are resized to the expected size
        model_transform = Compose(
            [
                Resize((self.default_image_size, self.default_image_size), interpolation=default_transform.transforms[0].interpolation),
                *default_transform.transforms[1:],
            ]
        )

        return featurizer, model_transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a combined FSDP wrapping policy to wrap ViT blocks, the full ViT, and SVD's UNet."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        svd_wrap_policy = partial(_module_wrap_policy, module_classes={SVDVisionBackbone})

        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy, svd_wrap_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run the transformed image tensors through their respective vision backbones and return the features.
        """
        all_patches = {}
        all_patches = {name: featurizer(pixel_values[name]) for name, featurizer in self.featurizers.items()}
        return all_patches

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """Return the default image resolution for SigLIP."""
        return self.first_vit_data_cfg["input_size"]

    @torch.no_grad()
    def preview(self, im_1_batched, output_dir="./wm_output"):
        """Function to preview the diffusion process by generating a GIF from the frames."""
        self.featurizers["svd"].preview(im_1_batched, output_dir=output_dir)

    @property
    def embed_dim(self) -> Dict[str, int]:
        """Return the embedding dimension for each featurizer."""
        dims = {}
        for name, featurizer in self.featurizers.items():
            dims[name] = featurizer.embed_dim
        return dims

    @property
    def num_patches(self) -> int:
        """Return the number of patches per model (they should be equal)."""
        # pick a non-svd key to get patch count
        other_key = next(key for key in self.featurizers if key != 'svd')
        self.other_key = other_key
        num_patches = self.featurizers[other_key].patch_embed.num_patches
        return num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Return the model's preferred half-precision floating type."""
        return torch.bfloat16