"""
siglipsvd.py

基于SigLIP和Stable Video Diffusion (SVD)的VisionBackbone，用于PrismaticVLM。
这个骨干网络将SVD的UNet中间层特征与SigLIP的特征融合作为视觉表征。
"""

from dataclasses import dataclass # type: ignore
from functools import partial # type: ignore
from typing import Callable, Dict, Tuple, Optional, List, Union, Any # type: ignore
import re # type: ignore
import ast # type: ignore

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                        transformer_auto_wrap_policy)
from torchvision.transforms import Compose, Resize
from torchvision import transforms
from torchvision.transforms import functional as TF

from prismatic.models.backbones.vision.base_vision import (ImageTransform,
                                                            VisionBackbone,
                                                            unpack_tuple)
from prismatic.models.backbones.vision.svd import SVDVisionBackbone
from prismatic.util.svd_utils import (customunet,
                                    process_anyres_image,
                                    unpad_image,
                                    get_anyres_image_grid_shape)


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
    "clip-svd_dual_siglip": {
        "siglip": "vit_so400m_patch14_siglip_224",
        "svd_height": 448,
        "svd_width": 448,
    },
    "clip-svd_dual_dino": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "svd_height": 448,
        "svd_width": 448,
    },
    "clip-svd_dual_clip": {
        "clip": "vit_large_patch14_clip_224.openai",
        "svd_height": 448,
        "svd_width": 448,
    },
}

# @dataclass
# class SiglipSVDImageTransform:
#     """一个数据类，用于封装SigLIP和SVD各自的图像变换。"""
#     transforms: Dict[str, ImageTransform]
#     is_prismatic: bool = True

#     def __call__(self, img: Union[PIL.Image.Image, List[PIL.Image.Image]], **kwargs: str) -> Dict[str, torch.Tensor]:
#         """对每个模型的变换都调用一次，返回一个包含所有结果的字典。"""
#         if isinstance(img, list):
#             transformed_img = {}
#             for name, transform in self.transforms.items():
#                 if name == "svd":
#                     transformed_img[name] = transform(img, **kwargs)
#                 else:
#                     transformed_img[name] = transform(img[0], **kwargs) 
#             return transformed_img
#         else:
#             return {
#                 name: transform(img, **kwargs)
#                 for name, transform in self.transforms.items()
#             }

def get_interpolation_mode_str(mode: InterpolationMode) -> str:
    if mode == InterpolationMode.BICUBIC:
        return 'bicubic'
    if mode == InterpolationMode.BILINEAR:
        return 'bilinear'
    if mode == InterpolationMode.NEAREST:
        return 'nearest'
    return 'bicubic' # 默认为bicubic，与timm常见配置一致

class BatchImageTransform(nn.Module):
    """
    一个高效的图像批处理变换模块。
    
    可以通过以下方式初始化:
    1. model_name: 方便的快速创建方式。
    2. data_config: 更高效、模块化的方式，避免重复解析。
    
    同时支持通过 `target_size` 覆盖默认的输出尺寸。
    """
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        data_config: Optional[Dict[str, Any]] = None, 
        target_size: Optional[int] = None
    ):
        super().__init__()
        
        # --- 步骤 1: 获取数据配置 (新逻辑) ---
        if data_config is None:
            if model_name is None:
                raise ValueError("必须提供 'model_name' 或 'data_config'。")
            model = timm.create_model(model_name, pretrained=False)
            self.data_config = timm.data.resolve_model_data_config(model)
            init_source = f"model_name '{model_name}'"
        else:
            self.data_config = data_config
            init_source = "a provided data_config"
        
        eval_transform = timm.data.create_transform(**self.data_config, is_training=False)
        
        self.resize_size = None
        self.interpolation_mode = 'bicubic'
        self.crop_size = None
        self.normalize = None

        for transform in eval_transform.transforms:
            if isinstance(transform, transforms.Resize):
                size = data_config["input_size"]
                self.resize_size = (size, size) if isinstance(size, int) else tuple(size)
                self.interpolation_mode = get_interpolation_mode_str(transform.interpolation)
            elif isinstance(transform, transforms.CenterCrop):
                size = transform.size
                self.crop_size = (size, size) if isinstance(size, int) else tuple(size)
            elif isinstance(transform, transforms.Normalize):
                self.normalize = transform

        if self.crop_size is None and self.resize_size is not None: self.crop_size = self.resize_size
        if self.normalize is None: raise ValueError(f"无法从配置中找到Normalize变换。")

        if target_size is not None:
            final_size = (target_size, target_size)
            logger.info(f"⚠️  Overriding detected size. Using target_size: {final_size}")
            self.resize_size = final_size

        logger.info(f"BatchImageTransform initialized from {init_source}:")
        logger.info(f"   - Resize to: {self.resize_size} (mode: {self.interpolation_mode})")
        logger.info(f"   - Center Crop to: {self.crop_size}")
        logger.info(f"   - Normalize Mean: {[f'{x:.4f}' for x in self.normalize.mean]}")
        logger.info(f"   - Normalize Std:  {[f'{x:.4f}' for x in self.normalize.std]}")

    @torch.inference_mode()
    def forward(self, pil_images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        '''
        Return: A tensor of shape [N C H W] where N is the number of images.
        '''
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        tensor_list = [TF.to_tensor(img) for img in pil_images]
        batch = torch.stack(tensor_list)
        if batch.shape[-2:] != self.resize_size:
            batch = F.interpolate(batch, size=self.resize_size, mode=self.interpolation_mode, antialias=True)
        _, _, H, W = batch.shape
        ch, cw = self.crop_size
        top = (H - ch) // 2
        left = (W - cw) // 2
        batch = batch[:, :, top:top+ch, left:left+cw]
        # print(f"Batch shape after resize and crop: {batch.shape}") N C H W
        return self.normalize(batch)


@dataclass
class SiglipSVDImageTransform:
    transforms: Dict[str, Any]
    resize_strategy: str
    patch_size: int
    grid_pinpoints: Optional[List[List[int]]] = None
    is_prismatic: bool = True

    def __call__(self, img: Union[PIL.Image.Image, List[PIL.Image.Image]], **kwargs: str) -> Dict[str, torch.Tensor]:
        # if isinstance(img, list):
        #     img_for_vit = img
        #     img_for_svd = img
        # else:
        img_for_vit = img
        img_for_svd = img

        transformed_img = {}
        for name, transform in self.transforms.items():
            if ( self.resize_strategy == "anyres" or "anyres_max" in self.resize_strategy) and name != "svd" :
                if not isinstance(img_for_vit, list):
                    img_for_vit = [img_for_vit]
                    
                img_tensor_list = []
                # logger.info(f"Processing {len(img_for_vit)} images.")
                if len(img_for_vit) > 1:
                    # print("multiple images -> using BatchTransform")
                    for img in img_for_vit:
                        # Shape Before BatchTransform: List[H W C]
                        # Shape After BatchTransform: [N C H W], in this case -> 1 C H W
                        # img = img.resize((336, 336))
                        original_img_patch = transform(img)
                        
                        # [DEBUG]: Shape: num_patches * c * h * w
                        img_tensor_list.append(original_img_patch)
                        # img shape: H W C
                        # img_tensor shape: C H W
                        # image_tensor = transform(img) # transforms.ToTensor(img)
                        # img_tensor_list.append(image_tensor.unsqueeze(0))
                    final_patches = torch.cat(img_tensor_list, dim=0) # N C H W
                    # img_tensor_list = [final_patches]
                    transformed_img[name] = final_patches # torch.stack(img_tensor_list, dim=0)
                    # print(f"Transformed image shape: {transformed_img[name].shape}")
                else:
                    # print("single image -> using AnyRes")
                    img = img_for_vit[0]
                    # img = img.resize((336, 336))
                    original_img_patch = transform(img)
                    
                    patches = process_anyres_image(img, self.patch_size, self.grid_pinpoints)
                    image_patches = transform(patches)
                    
                    final_patches = torch.cat([original_img_patch, image_patches], dim=0) 
                    # img_tensor_list.append(final_patches)
                    transformed_img[name] = final_patches # torch.stack(img_tensor_list, dim=0)
                # print(f"{name} transformed image shape: {transformed_img[name].shape}")
                
            else:
                if name == "svd":
                    if not isinstance(img_for_svd, list):
                        img_for_svd = [img_for_svd]
                    ts_img = transform(img_for_svd, **kwargs)
                    transformed_img[name] = ts_img # 1 C H W
                    # print(f"SVD transformed image shape: {transformed_img[name].shape}")
                else:
                    if isinstance(img_for_vit, list):
                        # logger.info(f"Before squueze shape: {transform(img_for_vit[0], **kwargs).shape}")
                        ts_img = transform(img_for_vit[0], **kwargs)
                        transformed_img[name] = ts_img.squeeze(0)
                        # logger.info(f"Transformed image shape for {name}: {transformed_img[name].shape}")
                    else:
                        transformed_img[name] = transform(img_for_vit, **kwargs).squeeze(0)
                        # logger.info(f"Transformed image shape for {name}: {transformed_img[name].shape}")
        return transformed_img
        
SUPPORTED_VIT_MODELS = ["dino", "siglip", "clip"]

class SVDSigLIPVisionBackbone(VisionBackbone):
    """
    一个结合了SigLIP和SVD特征的Vision Backbone。
    它会并行处理图像，然后拼接两个模型提取的特征。
    """
    def __init__(self, 
                vision_backbone_id: str, 
                image_resize_strategy: str, 
                default_image_size: int = 224,
                num_frames: Optional[int] = 14,
                image_grid_pinpoints: str = "(1, 6)x(1, 6)") -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        self.vision_backbone_id = vision_backbone_id
        if self.vision_backbone_id not in SIGLIPSVD_VISION_BACKBONES:
            raise ValueError(f"Vision backbone ID `{self.vision_backbone_id}` is not defined in SIGLIPSVD_VISION_BACKBONES.")

        model_cfg = SIGLIPSVD_VISION_BACKBONES[self.vision_backbone_id]
        svd_height = model_cfg["svd_height"]
        svd_width = model_cfg["svd_width"]
        
        # SVDVisionBackbone只支持 "resize-naive" 策略，因此组合模型也必须遵循此策略
        if self.image_resize_strategy not in ['resize-naive', 'anyres'] and 'anyres_max' not in self.image_resize_strategy:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported.")

        if self.image_resize_strategy == "anyres" or "anyres_max" in self.image_resize_strategy:
            patch_size = self.default_image_size
            
            if isinstance(image_grid_pinpoints, str) and "x" in image_grid_pinpoints:
                try:
                    matches = re.findall(r"\((\d+),\s*(\d+)\)x\((\d+),\s*(\d+)\)", image_grid_pinpoints)
                    start_w, end_w, start_h, end_h = map(int, matches[0])
                    generated_configs = [
                        (i, j) for i in range(start_w, end_w + 1) for j in range(start_h, end_h + 1)
                    ]
                    self.grid_pinpoints = [[w * patch_size, h * patch_size] for w, h in generated_configs]
                    print(f"Generated `anyres` grid with {self.grid_pinpoints} resolutions from string '{image_grid_pinpoints}'")
                    # logger.info(f"Generated `anyres` grid with {len(self.grid_pinpoints)} resolutions from string '{image_grid_pinpoints}'")
                except Exception:
                    raise ValueError(f"Could not parse grid string '{image_grid_pinpoints}'. Expected format '(start_w,end_w)x(start_h,end_h)'.")
            else:
                self.grid_pinpoints = ast.literal_eval(image_grid_pinpoints)
        else:
            self.grid_pinpoints = None

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
                if model_type == "siglip":
                    self.hidden_size = 1152
                elif model_type == "clip":
                    self.hidden_size = 768
                # TODO: DINO has no attribute 'hidden_size', create fake ones
                elif model_cfg == "dino":
                    self.hidden_size = 1024
                featurizer, transform = self._create_vision_encoder(model_cfg, model_type)
                self.featurizers[model_type] = featurizer
                image_transforms[model_type] = transform
                self.vit_component_keys.append(model_type)
        if not self.vit_component_keys:
            raise ValueError(f"Configuration for `{self.vision_backbone_id}` must include at least one of {SUPPORTED_VIT_MODELS}.")

        # 2. 初始化 SVD Featurizer
        svd_featurizer = SVDVisionBackbone(
            vision_backbone_id="clip-svd",  # 内部ID，不会被外部直接使用
            image_resize_strategy="resize-naive",
            height=svd_height,
            width=svd_width,
            num_frames=num_frames,
        )
        svd_featurizer.eval()
        self.featurizers["svd"] = svd_featurizer
        image_transforms["svd"] = self.featurizers["svd"].get_image_transform()

        # 3. 为两个模型创建图像变换
        self.image_transform = SiglipSVDImageTransform(
            transforms=image_transforms,
            resize_strategy=self.image_resize_strategy,
            patch_size=self.default_image_size,
            grid_pinpoints=self.grid_pinpoints,
        )
        
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
        self.total_num_patches = featurizer.patch_embed.num_patches
        featurizer.eval()

        # 修改forward方法，使其返回倒数第二层的特征
        # 注意: `get_intermediate_layers` 对于timm >= 0.10.0, n={-2}是更现代的方式
        featurizer.forward = unpack_tuple(
            partial(featurizer.get_intermediate_layers, n={len(featurizer.blocks) - 2})
        )
        
        # 加载图像变换
        model_data_cfg = timm.data.resolve_model_data_config(featurizer)
        model_data_cfg["input_size"] = (self.default_image_size, self.default_image_size)
        # default_transform = timm.data.create_transform(**model_data_cfg, is_training=False)
        
        # assert isinstance(default_transform, Compose), "Unexpected `default_image_transform`!"
        # assert isinstance(default_transform.transforms[0], Resize)

        # # 确保输入图像被调整到我们期望的大小
        # model_transform = Compose(
        #     [
        #         Resize((self.default_image_size, self.default_image_size), interpolation=default_transform.transforms[0].interpolation),
        #         *default_transform.transforms[1:],
        #     ]
        # )
        model_transform = BatchImageTransform(data_config=model_data_cfg)
        
        return featurizer, model_transform

    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回一个组合的FSDP封装策略，用于封装ViT的block、整个ViT以及SVD的UNet。"""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        svd_wrap_policy = partial(_module_wrap_policy, module_classes={SVDVisionBackbone})

        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy, svd_wrap_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor], image_sizes: Optional[List[Tuple[int, int]]] = None, real_lens: Optional[List[List[int]]] = None) -> Dict[str, torch.Tensor]:
        """
        通过各自的视觉骨干网络运行变换后的图像张量，并返回拼接后的patch特征。
        """
        embed_std = 1 / torch.sqrt(torch.tensor(self.hidden_size))
        image_newline = nn.Parameter(
            torch.randn(self.hidden_size) * embed_std
        )
        all_patches = {}
        # with torch.no_grad():
        for name, featurizer in self.featurizers.items():
            inputs = pixel_values[name]
            # print(f"Input pixel values shape for {name}: {inputs.shape}")
            if inputs.ndim == 5:
                # Inputs: B N C H W
                if name != "svd":
                    batch_new_image_features = []
                    for i, input_tensor in enumerate(inputs):
                        # inputs = inputs.squeeze(0)
                        if real_lens:
                            real_len = real_lens[i]
                            input_tensor = input_tensor[:real_len] # P C H W
                        sizes = image_sizes[i] # [(336, 336)]
                        # input_list = list(torch.unbind(input_tensor, dim=0)) # NOTE: MMSI SOTA
                        # NOTE: Added after SOTA
                        images = [input_tensor]
                        
                        images_list = []
                        for image in images:
                            if image.ndim == 4:
                                images_list.append(image)
                            else:
                                images_list.append(image.unsqueeze(0))

                        concat_images = input_tensor # torch.stack([image for image in images_list], dim=0)
                        split_sizes = [image.shape[0] for image in images_list]
                        # split_sizes = [1 for tens in input_tensor] # NOTE: MMSI SOTA
                        # Featurizer Input:  B * C * H * W
                        encoded_image_features = featurizer(concat_images) # -> Output: [(\Sigma(B_i)), num_tokens, embed_dim]
                        encoded_image_features = torch.split(encoded_image_features, split_sizes)
                        # logger.info(f"First Splited Feature Size: {encoded_image_features[0].shape}, num features: {len(encoded_image_features)}")
                        # NOTE: Added after SOTA
                        image_features = []
                        for idx, image_feat in enumerate(encoded_image_features):
                            image_features.append(image_feat)
                            
                        new_image_features = []
                        # print(f"sizes = {sizes}")
                        # print(f"Image features length for {name}: {len(image_features)}, each shape: {[feat.shape for feat in image_features]}")
                        # image_features shape: 
                        for image_idx, image_feature in enumerate(image_features):
                            # print(f"Image Feature Shape: {image_feature.shape}")
                            if image_feature.shape[0] > 1:  # multi patches and multi images operations
                                # logger.info(f"Processing multi images.")
                                # WARNING
                                base_image_feature = image_feature[0]
                                image_feature = image_feature[1:]
                                height = width = 224 // 14 # int(self.total_num_patches ** 0.5)
                                # logger.info(f"NumPatches: {self.total_num_patches}, Height = Width = {height}, Image Shape = {base_image_feature.shape}")
                                # assert height * width == base_image_feature.shape[0]

                                if "anyres_max" in self.image_resize_strategy:
                                    matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", self.image_resize_strategy)
                                    if matched_anyres_max_num_patches:
                                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                                # print(f"Matched max patches: {max_num_patches}")

                                if self.image_resize_strategy == "anyres" or "anyres_max" in self.image_resize_strategy:
                                    vision_tower_image_size = self.default_image_size
                                
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(sizes[image_idx], self.grid_pinpoints, vision_tower_image_size)
                                    # import pdb; pdb.set_trace()
                                    image_feature = image_feature.reshape(num_patch_height, num_patch_width, height, width, -1)
                                    # print(f"Reshaped image feature shape: {image_feature.shape}")
                                else:
                                    image_feature = image_feature.reshape(2, 2, height, width, -1)

                                if "anyres_max" in self.image_resize_strategy and matched_anyres_max_num_patches:
                                    unit = image_feature.shape[2]
                                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                    image_feature = unpad_image(image_feature, sizes[image_idx])
                                    c, h, w = image_feature.shape
                                    import math
                                    times = math.sqrt(h * w / (max_num_patches * unit**2))
                                    if times > 1.1:
                                        image_feature = image_feature[None]
                                        image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                                    image_feature = torch.cat((image_feature, image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                                new_image_features.append(image_feature)
                            else:  # single image operations
                                # print(f"Processing single image.")
                                image_feature = image_feature[0]
                                current_device = image_feature.device
                                image_feature = torch.cat(
                                    (image_feature, image_newline[None].to(current_device)), 
                                    dim=0
                                )
                                # image_feature = torch.cat((image_feature, image_newline[None]), dim=0)
                                new_image_features.append(image_feature)
                        batch_new_image_features.append(torch.cat(new_image_features, dim=0).unsqueeze(0))
                    concatenated_features = torch.cat(batch_new_image_features, dim=0)
                    # print(f"Concatenated features shape for {name}: {concatenated_features.shape}")
                    image_features = concatenated_features
                else:
                    image_features = []
                    for i, input_tensor in enumerate(inputs):
                        # inputs = inputs.squeeze(0)
                        if input_tensor.ndim == 3:
                            input_tensor = input_tensor.unsqueeze(0)
                        if real_lens:
                            real_len = real_lens[i]
                            input_tensor = input_tensor[:real_len]
                        image_feature = featurizer(input_tensor) # Output Shape: [B, N, D]
                        image_features.append(image_feature.reshape(-1, image_feature.shape[-1]))
                    concatenated_features = torch.stack(image_features, dim=0)
                    # print(f"Concatenated features shape for SVD: {concatenated_features.shape}")
                    image_features = concatenated_features
            else:
                # logger.info(f"Encoding Images Shape: {inputs.shape}")
                image_features = featurizer(inputs)
                # logger.info(f"Encoded image features shape: {image_features.shape}")
            all_patches[name] = image_features

        return all_patches

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """返回SigLIP的默认图像分辨率。"""
        return self.first_vit_data_cfg["input_size"]

    @property
    def embed_dim(self) -> Dict[str, int]:
        """返回拼接后总的特征维度。"""
        dims = {}
        for name, featurizer in self.featurizers.items():
            dims[name] = featurizer.embed_dim
        return dims


    @property
    def num_patches(self) -> int:
        """返回每个模型的patch数量（两者应相等）。"""
        # siglip_num_patches = self.siglip_featurizer.patch_embed.num_patches
        other_key = next(key for key in self.featurizers if key != 'svd')
        self.other_key = other_key
        num_patches = self.featurizers[other_key].patch_embed.num_patches

        # 添加断言以确保两个模型的patch数量一致，这是拼接操作的前提
        # assert siglip_num_patches == svd_num_patches, (
        #     f"Number of patches must match for concatenation! "
        #     f"SigLIP: {siglip_num_patches}, SVD: {svd_num_patches}"
        # )
        return num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """返回模型期望的半精度浮点类型。"""
        return torch.bfloat16