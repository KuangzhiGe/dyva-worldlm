"""
siglipsvd.py

基于SigLIP和Stable Video Diffusion (SVD)的VisionBackbone，用于PrismaticVLM。
这个骨干网络将SVD的UNet中间层特征与SigLIP的特征融合作为视觉表征。
"""

from dataclasses import dataclass # type: ignore
from functools import partial # type: ignore
from typing import Callable, Dict, Tuple, Optional, List, Union # type: ignore

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                        transformer_auto_wrap_policy)
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import (ImageTransform,
                                                            VisionBackbone,
                                                            unpack_tuple)
from prismatic.models.backbones.vision.svd import (SVDVisionBackbone,
                                                    customunet)


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


# Registry =>> Supported SigLIP-SVD Pairs
# 定义了SigLIP模型标识符以及SVD featurizer的输入图像尺寸
SIGLIPSVD_VISION_BACKBONES = {
    "clip-svd_dual_dino_siglip": {
        "siglip": "vit_so400m_patch14_siglip_224",
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "svd_height": 448,
        "svd_width": 448,
    },
}

@dataclass
class SiglipSVDImageTransform:
    """一个数据类，用于封装SigLIP和SVD各自的图像变换。"""
    transforms: Dict[str, ImageTransform]
    is_prismatic: bool = True

    def __call__(self, img: Union[Image, List[Image]], **kwargs: str) -> Dict[str, torch.Tensor]:
        """对每个模型的变换都调用一次，返回一个包含所有结果的字典。"""
        if isinstance(img, list):
            transformed_img = {}
            for name, transform in self.transforms.items():
                if name == "svd":
                    transformed_img[name] = transform(img, **kwargs)
                else:
                    transformed_img[name] = transform(img[0], **kwargs) 
            return transformed_img
        else:
            return {
                name: transform(img, **kwargs)
                for name, transform in self.transforms.items()
            }
        
SUPPORTED_VIT_MODELS = ["dino", "siglip", "clip"]

class DualSVDSigLIPDINOVisionBackbone(VisionBackbone):
    """
    一个结合了SigLIP和SVD特征的Vision Backbone。
    它会并行处理图像，然后拼接两个模型提取的特征。
    """
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.vision_backbone_id = vision_backbone_id
        if self.vision_backbone_id not in SIGLIPSVD_VISION_BACKBONES:
            raise ValueError(f"Vision backbone ID `{self.vision_backbone_id}` is not defined in SIGLIPSVD_VISION_BACKBONES.")

        model_cfg = SIGLIPSVD_VISION_BACKBONES[self.vision_backbone_id]
        svd_height = model_cfg["svd_height"]
        svd_width = model_cfg["svd_width"]
        
        # SVDVisionBackbone只支持 "resize-naive" 策略，因此组合模型也必须遵循此策略
        if self.image_resize_strategy != "resize-naive":
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported for SiglipSVD. "
                            "Only 'resize-naive' is supported due to SVD backbone limitations.")

        # 1. 初始化 SigLIP Featurizer (ViT)
        self.featurizers = nn.ModuleDict()
        image_transforms = {}
        
        # 2. 动态初始化所有在配置中指定的 ViT 模型
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

        # 2. 初始化 SVD Featurizer
        svd_featurizer = SVDVisionBackbone(
            vision_backbone_id="clip-svd",  # 内部ID，不会被外部直接使用
            image_resize_strategy=self.image_resize_strategy,
            height=svd_height,
            width=svd_width,
        )
        svd_featurizer.eval()
        self.featurizers["svd"] = svd_featurizer
        image_transforms["svd"] = self.featurizers["svd"].get_image_transform()

        # 3. 为两个模型创建图像变换
        self.image_transform = SiglipSVDImageTransform(transforms=image_transforms)
        
        # 4. 存储第一个ViT模型的数据配置，用于默认分辨率
        self.first_vit_key = self.vit_component_keys[0]
        self.first_vit_data_cfg = timm.data.resolve_model_data_config(self.featurizers[self.first_vit_key])
        self.first_vit_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)
        
    @retry(
        retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            f"加载模型失败，原因: {retry_state.outcome.exception()}. "
            f"将在 {retry_state.next_action.sleep:.2f} 秒后进行第 {retry_state.attempt_number} 次重试..."
        )
    )
    
    def _create_model_with_retry(self, url: str) -> VisionTransformer:
        """使用tenacity进行带重试的模型创建。"""
        model: VisionTransformer = timm.create_model(
            url,
            pretrained=True,
            num_classes=0,
            img_size=self.default_image_size
        )
        return model
    
    def _create_vision_encoder(self, model_cfg: Dict, model_type: str) -> Tuple[VisionTransformer, ImageTransform]:
        """
        根据给定的模型类型(siglip/dino/clip)创建视觉骨干和图像变换。
        """
        logger.info(f"Creating Vision Encoder for `{model_type}`...")
        
        # 加载模型
        timm_path_or_url = model_cfg[model_type]
        featurizer = self._create_model_with_retry(timm_path_or_url)
        featurizer.eval()

        # 修改forward方法，使其返回倒数第二层的特征
        # 注意: `get_intermediate_layers` 对于timm >= 0.10.0, n={-2}是更现代的方式
        featurizer.forward = unpack_tuple(
            partial(featurizer.get_intermediate_layers, n={len(featurizer.blocks) - 2})
        )
        
        # 加载图像变换
        model_data_cfg = timm.data.resolve_model_data_config(featurizer)
        model_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)
        default_transform = timm.data.create_transform(**model_data_cfg, is_training=False)
        
        assert isinstance(default_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_transform.transforms[0], Resize)

        # 确保输入图像被调整到我们期望的大小
        model_transform = Compose(
            [
                Resize((self.default_image_size, self.default_image_size), interpolation=default_transform.transforms[0].interpolation),
                *default_transform.transforms[1:],
            ]
        )
        
        return featurizer, model_transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回一个组合的FSDP封装策略，用于封装ViT的block、整个ViT以及SVD的UNet。"""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        svd_wrap_policy = partial(_module_wrap_policy, module_classes={SVDVisionBackbone})

        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy, svd_wrap_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        通过各自的视觉骨干网络运行变换后的图像张量，并返回拼接后的patch特征。
        """
        all_patches = {}
        # with torch.no_grad():
        # for name, featurizer in self.featurizers.items():
        #     # 从输入字典中获取对应模型的像素值
        #     patches = featurizer(pixel_values[name])
        #     all_patches[name] = patches
        siglip_patches = self.featurizers["siglip"](pixel_values["siglip"])
        dino_patches = self.featurizers["dino"](pixel_values["dino"])
        svd_patches = self.featurizers["svd"](pixel_values["svd"])
        dino_siglip_patches = torch.cat([siglip_patches, dino_patches], dim=2)
        all_patches["dino_siglip"] = dino_siglip_patches
        all_patches["svd"] = svd_patches

        return all_patches

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """返回SigLIP的默认图像分辨率。"""
        return self.first_vit_data_cfg["input_size"]

    @property
    def embed_dim(self) -> Dict[str, int]:
        """返回拼接后总的特征维度。"""
        dims = {}
        
        dims["dino_siglip"] = self.featurizers["siglip"].embed_dim + self.featurizers["dino"].embed_dim
        dims["svd"] = self.featurizers["svd"].embed_dim
        return dims
        

    @property
    def num_patches(self) -> int:
        """返回每个模型的patch数量（两者应相等）。"""
        # siglip_num_patches = self.siglip_featurizer.patch_embed.num_patches
        svd_num_patches = self.featurizer["svd"].num_patches
        
        # 添加断言以确保两个模型的patch数量一致，这是拼接操作的前提
        # assert siglip_num_patches == svd_num_patches, (
        #     f"Number of patches must match for concatenation! "
        #     f"SigLIP: {siglip_num_patches}, SVD: {svd_num_patches}"
        # )
        return svd_num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """返回模型期望的半精度浮点类型。"""
        return torch.bfloat16