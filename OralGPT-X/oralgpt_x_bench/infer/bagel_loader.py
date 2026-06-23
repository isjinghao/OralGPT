"""Load BAGEL checkpoint for unified_edit inference without modifying Bagel source."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


@dataclass
class BagelEditRuntime:
    model: Bagel
    vae_model: torch.nn.Module
    tokenizer: Qwen2Tokenizer
    new_token_ids: dict
    vae_transform: ImageTransform
    vit_transform: ImageTransform
    device: str


def load_bagel_for_edit(
    model_path: str,
    device: str,
    max_latent_size: int = 64,
) -> BagelEditRuntime:
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
    model.load_state_dict(model_state_dict, strict=False)
    del model_state_dict

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()

    return BagelEditRuntime(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 378, 14),
        device=device,
    )
