"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    is_anyres: bool = False

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def _pad_to_max_p(self, tensor: torch.Tensor, max_p: int = 36) -> torch.Tensor:
        p_dim = tensor.shape[0]
        if p_dim == max_p:
            return tensor
        if p_dim < max_p:
            _, C, H, W = tensor.shape
            padding_size = max_p - p_dim
            padding = torch.zeros(padding_size, C, H, W, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        else:
            return tensor[:max_p, ...]

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx, pv in enumerate(pixel_values) if pv is not None], dtype=torch.long
        )
        # multimodal_indices = torch.tensor(
        #     [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        # )
        real_lens = None
        image_sizes = None
        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            # Find the max number of images in the batch to pad to
            max_num_images = max(pv.shape[0] for pv in pixel_values if pv is not None)
            
            # Pad or use dummy for each example
            padded_pixel_values = []
            for idx in range(len(input_ids)):
                if idx in multimodal_indices:
                    current_images = pixel_values[idx]
                    num_images = current_images.shape[0]
                    
                    if num_images < max_num_images:
                        # Create padding tensor and concatenate
                        padding = torch.zeros(
                            (max_num_images - num_images,) + current_images.shape[1:],
                            dtype=current_images.dtype,
                            device=current_images.device
                        )
                        padded_images = torch.cat([current_images, padding], dim=0)
                        padded_pixel_values.append(padded_images)
                    else:
                        padded_pixel_values.append(current_images)
                else:
                    # For non-multimodal samples, pad with dummy images to match max_num_images
                    dummy_padding = self.dummy_pixel_values.unsqueeze(0).repeat(max_num_images, 1, 1, 1)
                    padded_pixel_values.append(dummy_padding)

            # Stack the now-uniform list of tensors
            # This is the single place where torch.stack is called for pixel values
            pixel_values = torch.stack(padded_pixel_values, dim=0)
            # pixel_values = torch.stack(
            #     [
            #         pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
            #         for idx in range(len(input_ids))
            #     ]
            # )
        elif isinstance(pv_example, dict):
            if not self.is_anyres:
                pixel_values = {
                    k: torch.stack(
                        [
                            pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                            for idx in range(len(input_ids))
                        ]
                    )
                    for k in pv_example
                }
            else:
                image_sizes = [instance["image_sizes"] for instance in instances]
                for k in pv_example:
                    real_lens = [pixel_values[idx][k].shape[0] if idx in multimodal_indices else self.dummy_pixel_values.shape[0] for idx in range(len(input_ids))]
                    break  # just need to do this once
                max_p_in_batch = max(real_lens)
                pixel_values = {
                    k: torch.stack(
                        [
                            self._pad_to_max_p(pixel_values[idx][k], max_p=max_p_in_batch) if idx in multimodal_indices else self._pad_to_max_p(self.dummy_pixel_values, max_p=max_p_in_batch)
                            for idx in range(len(input_ids))
                        ]
                    )
                    for k in pv_example
                }
                # pixel_values = {
                #     k: [
                #         pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                #         for idx in range(len(input_ids))
                #     ]
                # }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
            image_sizes=image_sizes if image_sizes else None,
            real_lens=real_lens if real_lens else None,
        )
