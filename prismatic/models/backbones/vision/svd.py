"""
svd.py

VisionBackbone based on Stable Video Diffusion (SVD) for PrismaticVLM.
This backbone uses the intermediate features from SVD's UNet as visual representations.
"""

from dataclasses import dataclass # type: ignore
from functools import partial # type: ignore
from typing import Callable, Tuple, Union, Dict, Optional, List # type: ignore
import re # type: ignore
import os

import torch
import torch.nn as nn
from torchvision import transforms
import PIL
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from diffusers import StableVideoDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

import timm
from timm.models.vision_transformer import Block, VisionTransformer

from prismatic.util.svd_utils import randn_tensor, retrieve_timesteps, _resize_with_antialiasing, _append_dims, customunet, SVDImageTransform, handle_memory_attention
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

class SVDVisionBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        height: Optional[int] = 448,
        width: Optional[int] = 448,
        num_frames: Optional[int] = 8,
        # SVD 相关路径
        svd_model_path: str = "/mnt/world_foundational_model/kevin/vlm_models/svd/",
        clip_model_path: str = "/mnt/world_foundational_model/kevin/vlm_models/clip-vit-base-patch32",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)

        self.svd_model_path = svd_model_path
        self.clip_model_path = clip_model_path
        self.torch_dtype = torch_dtype
        self.height = height if height is not None else 448
        self.width = width if width is not None else 448
        self.num_frames = num_frames if num_frames is not None else 8
        # overwatch.info(f"Initialized SVDVisionBackbone with height: {self.height}, width: {self.width}, num_frames: {self.num_frames}")

        # 1. 加载SVD pipeline和相关模型
        self.pipeline, self.vae, self.unet = self._load_svd_models()
        self.image_encoder = self._load_image_encoder()
        self.vae_scale_factor = self.pipeline.vae_scale_factor

        handle_memory_attention(
            enable_xformers_memory_efficient_attention=False,
            enable_torch_2_attn=True,
            unet=self.unet
        )

        # 2. 设置 image transform
        self.image_transform =self. _create_image_transform() # svd_transform

        # 3. 将模型移至适当的设备和数据类型
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
        # new_unet.to(dtype=pipeline.unet.dtype, device=pipeline.unet.device)
        new_unet.to(dtype=self.torch_dtype)
        pipeline.unet = new_unet

        if pipeline.vae is not None:
            pipeline.vae.to(dtype=self.torch_dtype)

        # print("!!! Applying the critical fix: Switching UNet to the default attention processor.")
        # new_unet.set_default_attn_processor()

        return pipeline, pipeline.vae, pipeline.unet

    def _load_image_encoder(self):
        """加载CLIP图像编码器。"""
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_path)
        image_encoder.to(dtype=self.torch_dtype)
        image_encoder.requires_grad_(False)
        return image_encoder

    def _create_image_transform(self) -> Callable:
        """根据策略创建图像变换。"""
        # SVD对输入尺寸有特定要求，这里我们使用简单的resize
        # 你可以根据需要实现 'resize-crop' 和 'letterbox'
        if self.image_resize_strategy == "resize-naive":
            # 1. 首先定义针对单帧图像的变换
            single_frame_transform = Compose([
                Resize((self.height, self.width)),
                ToTensor(),
            ])

            # 2. 将单帧变换包装在我们的多帧处理器中
            return SVDImageTransform(single_frame_transform)
        else:
            # 为其他策略提供一个默认或抛出错误
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported for SVD!")

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        通过ddim_one_step运行图像，提取并投影特征。
        Args:
            pixel_values (torch.Tensor): 经过预处理的图像张量，形状为 [B, N, C, H, W]。
        Returns:
            torch.Tensor: 提取的视觉特征，形状为 [B, NumPatches, EmbedDim]。
        """
        # 将模型移动到与输入相同的设备
        self.pipeline.to(pixel_values.device)
        self.image_encoder.to(pixel_values.device)

        is_multiframe = pixel_values.shape[1] > 1
        conditioning_image = pixel_values[:, 0]
        if is_multiframe:
            latents = self.ddim_one_step(
                image=conditioning_image,  # 使用第一帧进行条件编码
                pipeline=self.pipeline,
                vae=self.vae,
                unet=self.unet,
                image_encoder=self.image_encoder,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                output_type="unet_latent",
                all_frames_pixels=pixel_values,
            )
        else:
            latents = self.ddim_one_step(
                image=conditioning_image,  # 使用第一帧进行条件编码
                pipeline=self.pipeline,
                vae=self.vae,
                unet=self.unet,
                image_encoder=self.image_encoder,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                output_type="unet_latent",
            )
        # B=batch_size, T=num_frames, C=channels, H_feat,W_feat=feature map size

        # 展平并投影特征
        # rearrange(latents, 'b t c h w -> b (t h w) c')  <- 注意维度顺序
        latents_flat = latents.permute(0, 1, 3, 4, 2).reshape(latents.size(0), -1, latents.size(2))

        return latents_flat

    @torch.no_grad()
    def preview(self, im_1_batched, output_dir="./wm_output"):
        """ Function to preview the diffusion process by generating a GIF from the frames."""
        is_multiframe = im_1_batched.shape[1] > 1
        # print(f"is_multiframe: {is_multiframe}")
        conditioning_image = im_1_batched[:, 0]
        if is_multiframe:
            frames = self.ddim_one_step(conditioning_image, self.pipeline, self.vae, self.unet, self.image_encoder, height=self.height,width=self.width, output_type="pil", all_frames_pixels=im_1_batched, num_frames=8)  # returning frames
            fully_denoised_frames = self.ddim_one_step(conditioning_image, self.pipeline, self.vae, self.unet, self.image_encoder, height=self.height,width=self.width, output_type="pil", all_frames_pixels=im_1_batched, num_frames=8, num_inference_steps=50)  # returning frames
        else:
            frames = self.ddim_one_step(conditioning_image, self.pipeline, self.vae, self.unet, self.image_encoder, height=self.height,width=self.width, output_type="pil", num_frames=8)  # returning frames
            fully_denoised_frames = self.ddim_one_step(conditioning_image, self.pipeline, self.vae, self.unet, self.image_encoder, height=self.height,width=self.width, output_type="pil", num_frames=8, num_inference_steps=50)  # returning frames
        # Create a directory to save the frames
        output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Convert frames to a GIF
        gif_path = os.path.join(output_dir, "output.gif")
        full_gif_path = os.path.join(output_dir, "output_full.gif")
        all_frames = []
        all_frames_full = []

        for batch_idx, batch in enumerate(frames):
            for frame_idx, frame in enumerate(batch):
                fully_denoised_frame = fully_denoised_frames[batch_idx][frame_idx]
                if isinstance(frame, torch.Tensor):  # Convert tensor to PIL Image if needed
                    frame = transforms.ToPILImage()(frame.cpu())
                if isinstance(fully_denoised_frame, torch.Tensor):
                    fully_denoised_frame = transforms.ToPILImage()(fully_denoised_frame.cpu())
                all_frames.append(frame)
                all_frames_full.append(fully_denoised_frame)

        # Save frames as a GIF
        if all_frames:
            all_frames[0].save(
                gif_path,
                save_all=True,
                append_images=all_frames[1:],
                duration=100,  # Duration in milliseconds per frame
                loop=0  # Loop forever
            )
            all_frames_full[0].save(
                full_gif_path,
                save_all=True,
                append_images=all_frames_full[1:],
                duration=100,  # Duration in milliseconds per frame
                loop=0  # Loop forever
            )
        # Save all the frames individually
        if all_frames:
            for idx, frame in enumerate(all_frames):
                frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
                full_frame_path = os.path.join(output_dir, f"full_frame_{idx:03d}.png")
                frame.save(frame_path)
                all_frames_full[idx].save(full_frame_path)
                # print(f"Saved frame {idx} to {frame_path}")

        # first_frame_path = os.path.join(output_dir, "first_frame.png")
        # if all_frames:
        #     first_frame = all_frames[0]
        #     first_frame.save(first_frame_path)
        #     print(f"First frame saved to {first_frame_path}")

        print(f"GIF saved to {gif_path}")

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
        all_frames_pixels: Optional[torch.Tensor] = None,
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
        # print("shape of the image to vae:", image.shape)
        image_latents = pipeline._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        # print("shape of the image latents:", image_latents.shape)
        # image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        # if needs_upcasting:
        #     pipeline.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        if all_frames_pixels is not None and all_frames_pixels.shape[1] > 0:
            num_input_frames = all_frames_pixels.shape[1]

            all_frames_pixels_flat = all_frames_pixels.flatten(0, 1)

            batch_pils = tensor_to_PIL(all_frames_pixels_flat) if isinstance(all_frames_pixels_flat, torch.Tensor) else all_frames_pixels_flat # input is 4 dim

            input_frames = pipeline.video_processor.preprocess(batch_pils, height=height, width=width).to(device=device, dtype=dtype)
            input_frames = input_frames + noise_aug_strength * randn_tensor(input_frames.shape, generator=generator, device=device, dtype=input_frames.dtype)

            input_frames_latents = pipeline._encode_vae_image(
                input_frames,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            ) # [B * num_frames, c, h, w]

            image_latents_with_frames = input_frames_latents.unflatten(0, (batch_size*2, num_input_frames))

            indices_to_replace = torch.linspace(0, num_frames - 1, steps=num_input_frames).round().long()
            image_latents[:, indices_to_replace] = image_latents_with_frames


        # 5. Get Added Time IDs
        added_time_ids = pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, None, sigmas)
        timesteps.to(device, dtype=dtype)
        # 7. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        # print(latents.shape, image_latents.shape)

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        pipeline._guidance_scale = guidance_scale
        # img_cognition_features = self.image_compressor(img_emb)

        # x_start = z.detach().clone()
        # noise = torch.randn_like(x_start)
        # fix_video_timesteps=True
        # if fix_video_timesteps==True:
        #     t_vid = torch.randint(pipeline.fake_ddpm_step, pipeline.fake_ddpm_step+1,(x_start.shape[0],), device=device).long()

        # 9. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        pipeline._num_timesteps = len(timesteps)

        #print(f"Running {num_inference_steps} inference steps with {len(timesteps)} timesteps")
        for i, t in enumerate(timesteps[:]):
            #print(f"Running inference step {i + 1}/{num_inference_steps} at timestep {t.item()}")

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents # [bsz*2,frame,4,32,32] Doubling bsz for guidance
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            latent_model_input.to(dtype=latents.dtype)
            # print(latent_model_input.shape, image_latents.shape) # (bsz*2,frame,4,32,32) (bsz*2,frame,4,32,32)

            # predict the noise residual
            # latent_model_input = latent_model_input.to(pipeline.unet.dtype)
            # image_embeddings = image_embeddings.to(pipeline.unet.dtype)

            if output_type == "unet_latent":
                return_down_features = True  # We want the down features for unet_latent output
            else:
                return_down_features = False

            t.to(dtype=latent_model_input.dtype, device=latent_model_input.device)

            # print(f"latent_model_input shape {latent_model_input.shape}, image_embeddings shape: {image_embeddings.shape}")
            full_noise_pred, down_noise_pred = self.unet( #use custom unet forward
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
                return_down_features=return_down_features,  # We want the down features
            ) # Get the down features
            # if return_down_features:
            #     print(f"down noise predict in timestep {i}", down_noise_pred.shape)

            noise_pred = full_noise_pred

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if output_type != "unet_latent":
                latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            # if XLA_AVAILABLE:
            #         xm.mark_step()

        if output_type == "unet_latent":
            noise_pred_uncond, noise_pred_cond = down_noise_pred.chunk(2) # [2 * b, t, c, h, w] [2, 8, 1280, 9, 16], h w are feature map of middle unet layer
            # print(f"noise_pred_cond shape: {noise_pred_cond.shape}") #noise_pred_cond shape: torch.Size([1, 8, 1280, 7, 7])
            # prepare latents based on latent unet middle layer shape

            latents_middle = pipeline.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames=noise_pred_cond.shape[1],
                num_channels_latents= noise_pred_cond.shape[2] * 2,
                height=noise_pred_cond.shape[3] * pipeline.vae_scale_factor,
                width=noise_pred_cond.shape[4] * pipeline.vae_scale_factor,
                dtype=noise_pred_cond.dtype,
                device=noise_pred_cond.device,
                generator=generator,
                latents=None,
            )
            latents_middle = pipeline.scheduler.step(noise_pred_cond, t, latents_middle).prev_sample
            frames = latents_middle # [b, t, c, h, w] 1280, 4, 4
            frames = frames.to(dtype=dtype)
            pipeline.maybe_free_model_hooks()
            return frames
        elif not output_type == "latent":
            original_vae_dtype = pipeline.vae.dtype
            pipeline.vae.to(dtype=torch.float32)
            latents = latents.to(torch.float32)
            with torch.no_grad():
                frames = pipeline.decode_latents(latents, num_frames, decode_chunk_size)
                frames = pipeline.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents
            frames = frames.to(torch.float32)
        pipeline.maybe_free_model_hooks()
        return frames


    def get_fsdp_wrapping_policy(self) -> Callable:
        """为SVD UNet定义FSDP封装策略。"""
        return partial(_module_wrap_policy, module_classes={SVDVisionBackbone})

    @property
    def embed_dim(self) -> int:
        # 这是投影MLP之后的输出维度
        # 来自于你的 `init_projection`: 'hidden_size': 4096
        return 1280

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