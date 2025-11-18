"""
qwen25.py

Class definition for all LLMs derived from Qwen2ForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder, Qwen25ChatPromptBuilder

# Registry =>> Support Qwen 2.5 Models (from HF Transformers)
# fmt: off
QWEN25_MODELS = {
    "qwen25-7b-chat": {
        "llm_family": "qwen25", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B-Instruct"
    },
    "qwen25-7b-pure": {
        "llm_family": "qwen25", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B"
    },
    "qwen25-72b-chat": {
        "llm_family": "qwen25", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-72B-Instruct"
    },
    "qwen25-32b-chat": {
        "llm_family": "qwen25", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-32B-Instruct"
    },
}


class Qwen25LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN25_MODELS[llm_backbone_id],
        )

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.endswith("-pure"):
            return PurePromptBuilder
        elif self.identifier.endswith("-chat"):
            return Qwen25ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16