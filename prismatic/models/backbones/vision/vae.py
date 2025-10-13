"""
svd_vision_backbone.py

基于Stable Video Diffusion (SVD)的VisionBackbone，用于PrismaticVLM。
这个骨干网络将SVD的UNet中间层特征作为视觉表征。
"""

from dataclasses import dataclass # type: ignore
from functools import partial # type: ignore
from typing import Callable, Tuple, Union, Dict, Optional, List # type: ignore
import re # type: ignore
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
import PIL
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from diffusers import StableVideoDiffusionPipeline
from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.models import UNetSpatioTemporalConditionModel
from transformers import CLIPVisionModelWithProjection

import timm
from timm.models.vision_transformer import Block, VisionTransformer
# 假设你的 svd_utils.py 和原始代码在同一个地方
from prismatic.util.svd_utils import randn_tensor, retrieve_timesteps, _resize_with_antialiasing
from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple
from prismatic.overwatch import initialize_overwatch
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

overwatch = initialize_overwatch(__name__)

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

class customunet(UNetSpatioTemporalConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        return_down_features: bool = True,  # NEW: flag to return down features
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple, Dict]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
            return_down_features (`bool`, *optional*, defaults to `False`):
                If True, also return the features after the down blocks (before mid block).
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple` or `dict`:
                If `return_down_features` is True, returns a dict with keys 'sample' and 'down_features'.
                Otherwise, behaves as usual.
        """
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb
        # Flatten the batch and frames dimensions
        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
        )

        # 2. pre-process
        sample = self.conv_in(sample)
        
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # --- Extract features before mid block ---
        #TODO: Design consider whether or not keep residual connections, allowing skip to latent

        down_features = sample  # shape: [batch * frames, ...]
        # Optionally, you can reshape to [batch, num_frames, ...] if needed:
        down_features = down_features.view(batch_size, num_frames, *down_features.shape[1:])
        # ------------------------------------------------

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if return_down_features:
            if not return_dict:
                return (sample, down_features)
            return {"sample": sample, "down_features": down_features}

        if not return_dict:
            return (sample,None)

        return UNetSpatioTemporalConditionOutput(sample=sample), None

VAE_VISION_BACKBONES = {
    "clip-svd_vae": {
        "svd_height": 448,
        "svd_width": 448,
    },
}

class VAEVisionBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        height: Optional[int] = 448,
        width: Optional[int] = 448,
        # SVD 相关路径
        svd_model_path: str = "/mnt/world_foundational_model/kevin/vlm_models/svd/",
        clip_model_path: str = "/mnt/world_foundational_model/kevin/vlm_models/clip-vit-base-patch32",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)

        model_cfg = VAE_VISION_BACKBONES[vision_backbone_id]
        self.svd_model_path = svd_model_path
        self.clip_model_path = clip_model_path
        self.torch_dtype = torch_dtype
        self.height = height if height is not None else 256
        self.width = width if width is not None else 512

        # 1. 加载SVD pipeline和相关模型
        self.pipeline, self.vae, self.unet = self._load_svd_models()
        self.image_encoder = self._load_image_encoder()
        
        self.vae_scale_factor = self.pipeline.vae_scale_factor
        
        # 2. 初始化你的WM模型作为特征提取器
        # 我们将WM的功能直接整合到这个类中，而不是创建一个单独的WM实例
        self.featurizer = self.unet

        # 3. 设置 image transform
        svd_transform = self._create_image_transform()
        
        # self.image_transform = DinoSiglipSVDImageTransform(siglip_transform, dino_transform, svd_transform)
        self.image_transform =self. _create_image_transform() # svd_transform
        # 4. 将模型移至适当的设备和数据类型
        self.to(self.torch_dtype)
        if self.vae is not None:
            self.vae.config.force_upcast = False
    @retry(
        retry=retry_if_exception_type(NETWORK_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            f"加载模型失败，原因: {retry_state.outcome.exception()}. "
            f"将在 {retry_state.next_action.sleep:.2f} 秒后进行第 {retry_state.attempt_number} 次重试..."
        )
    )

    def _load_svd_models(self):
        """加载SVD pipeline, VAE, 和自定义的UNet。"""
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.svd_model_path,
            torch_dtype=self.torch_dtype,
            variant='fp16'
        )
        
        # 使用你的 customunet
        config = pipeline.unet.config
        state_dict = pipeline.unet.state_dict()
        new_unet = customunet(config)
        new_unet.load_state_dict(state_dict)
        new_unet.to(dtype=pipeline.unet.dtype, device=pipeline.unet.device)
        pipeline.unet = new_unet
        
        # print("!!! Applying the critical fix: Switching UNet to the default attention processor.")
        # new_unet.set_default_attn_processor()

        return pipeline, pipeline.vae, pipeline.unet

    def _load_image_encoder(self):
        """加载CLIP图像编码器。"""
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_path)
        image_encoder.requires_grad_(False)
        return image_encoder

    def _create_image_transform(self) -> ImageTransform:
        """根据策略创建图像变换。"""
        # SVD对输入尺寸有特定要求，这里我们使用简单的resize
        # 你可以根据需要实现 'resize-crop' 和 'letterbox'
        if self.image_resize_strategy == "resize-naive":
            return Compose([
                Resize((self.height, self.width)),
                ToTensor(),
                # Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ])
        else:
            # 为其他策略提供一个默认或抛出错误
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported for SVD!")

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        通过ddim_one_step运行图像，提取并投影特征。
        
        Args:
            pixel_values (torch.Tensor): 经过预处理的图像张量，形状为 [B, C, H, W]。

        Returns:
            torch.Tensor: 提取的视觉特征，形状为 [B, NumPatches, EmbedDim]。
        """
        # 将模型移动到与输入相同的设备
        self.pipeline.to(pixel_values.device)
        self.image_encoder.to(pixel_values.device)
        
        # 调用ddim_one_step以获取unet的中间层特征
        # `output_type="unet_latent"` 会触发返回 `down_noise_pred` 的逻辑
        latents = self.ddim_one_step(
            pixel_values, self.pipeline, self.vae, self.unet, self.image_encoder,
            height=self.height, width=self.width, output_type="unet_latent"
        )
        # print(f"Latents Shape: {latents.shape}")
        # latents_permuted = latents.permute(0, 2, 3, 1)
        # print (f"Permuted Latents Shape {latents_permuted.shape}")
        # # 然后，将空间维度 (H, W) 合并为一个维度 (NumPatches)
        # # [B, H, W, C] -> [B, H*W, C]
        # features = latents_permuted.reshape(latents.size(0), -1, latents.size(1))
        # latents_flat = latents.permute(0, 1, 3, 4, 2).reshape(latents.size(0), -1, latents.size(2))
        
        return latents
    
    @torch.no_grad()
    def ddim_one_step(self, image, pipeline, vae, unet, image_encoder,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = 8,
        num_inference_steps: int = 1, #1000
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        """
        model forward一次 ，取出一次forward的feature
        noise_shape = [args.bs, channels, n_frames, h, w]
        image [b,c,h,w] 
        video [b,c,t,h,w] 
        return:
            cognition_features: [B, T, D]
        """
        pipeline.vae.eval()
        image_encoder.eval()
        device = unet.device
        dtype = self.torch_dtype

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else unet.config.num_frames

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        pipeline._guidance_scale = max_guidance_scale

        # clip_image = _resize_with_antialiasing(image, (224, 224))
        
        
        do_classifier_free_guidance = True
        # 3. Encode input image
        # if isinstance(image, torch.Tensor) and image.shape != (224, 224):
        #     image = _resize_with_antialiasing(image, (224, 224))

        def tensor_to_PIL(image):
            """将 PyTorch 张量 (Batch) 转换为 PIL 图像列表"""
            from PIL import Image
            import numpy as np
            if not isinstance(image, torch.Tensor) or image.dim() != 4:
                raise TypeError("输入必须是一个4维的PyTorch张量 (B, C, H, W)")
            image = image.clamp(0, 1) * 255
            numpy_array = image.permute(0, 2, 3, 1).cpu().byte().numpy()
            pil_images = [Image.fromarray(img) for img in numpy_array]
            return pil_images
            
        image = tensor_to_PIL(image) if isinstance(image, torch.Tensor) else image
        
        def numpy_batch_to_pt(numpy_batch):
            """将 numpy 批处理数组 (B, H, W, C) 转换为 PyTorch 张量 (B, C, H, W)"""
            return torch.tensor(numpy_batch, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        if isinstance(image[0], PIL.Image.Image):
            # overwatch.info(f"Input is a list of PIL images, converting to tensors for CLIP Processor")
            pil_images = image.copy()
            import numpy as np
            numpy_arrays = [np.array(im) for im in pil_images]
            image_batch_np = np.stack(numpy_arrays, axis=0)
            image_batch_tensor = numpy_batch_to_pt(image_batch_np)
            image_batch_tensor = image_batch_tensor * 2.0 - 1.0
            image_batch_tensor = _resize_with_antialiasing(image_batch_tensor, (224, 224))
            image = (image_batch_tensor + 1.0) / 2.0
        else:
            pil_images = image

        pipeline.image_encoder.to(dtype=dtype, device=device)
        # NOTE: need to hack numpy to support bf16 "pt": lambda obj: obj.detach().cpu().float().numpy().astype(ml_dtypes.bfloat16),
        # In site-packages/transformers/utils/generic.py
        # https://github.com/huggingface/diffusers/issues/7598
        # workaround: https://github.com/pytorch/pytorch/issues/109873#issuecomment-2019226035
        image_embeddings = pipeline._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        # image_embeddings.to(dtype=dtype)
        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        image = pipeline.video_processor.preprocess(pil_images, height=height, width=width).to(device=device, dtype=dtype)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        # needs_upcasting = False #pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
        # if needs_upcasting:
        #     vae.to(dtype=torch.float32)

        pipeline.vae.to(dtype=dtype, device=device)
        image_latents = pipeline._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        # image_latents = image_latents.to(image_embeddings.dtype)

        return image_latents

    def get_fsdp_wrapping_policy(self) -> Callable:
        """为SVD UNet定义FSDP封装策略。"""
        # 这是一个示例策略，你可能需要根据UNet的具体结构进行调整
        # 这里我们封装整个UNet
        svd_wrap_policy = partial(_module_wrap_policy, module_classes={VAEVisionBackbone})
        return svd_wrap_policy
    
    @property
    def embed_dim(self) -> int:
        # 这是投影MLP之后的输出维度
        print(f"Embedding Dim = {self.unet.conv_in.in_channels}")
        return 4 # self.unet.conv_in.in_channels

    @property
    def num_patches(self) -> int:
        num_frames = 8 
        h_feat = self.height // self.vae_scale_factor
        w_feat = self.width // self.vae_scale_factor
        overwatch.info(f"num_patches: {num_frames * h_feat * w_feat}")
        return num_frames * h_feat * w_feat

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.height, self.width)

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.torch_dtype