"""Text-to-image inference for OralGPT-X-Bench (adapted from Bagel eval/gen/gen_images_mp.py)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from modeling.bagel.qwen2_navit import NaiveCache

from .bagel_loader import BagelEditRuntime


def _move_to_device(generation_input: dict, device: str) -> dict:
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            generation_input[key] = value.to(device)
    return generation_input


def run_t2i_inference(
    runtime: BagelEditRuntime,
    prompt: str,
    *,
    resolution: int = 512,
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_interval: list[float] | None = None,
    cfg_renorm_min: float = 0.0,
    timestep_shift: float = 3.0,
) -> Image.Image:
    if cfg_interval is None:
        cfg_interval = [0.0, 1.0]

    gen_model = runtime.model
    vae_model = runtime.vae_model
    tokenizer = runtime.tokenizer
    new_token_ids = runtime.new_token_ids
    device = runtime.device

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = _move_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)],
        new_token_ids=new_token_ids,
    )
    generation_input = _move_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens, cfg_new_rope = [0], [0]
    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)],
    )
    generation_input_cfg = _move_to_device(generation_input_cfg, device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type="global",
            timestep_shift=timestep_shift,
            cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
            **generation_input,
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, resolution // 16, resolution // 16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, resolution // 8, resolution // 8)
    vae_dtype = next(vae_model.parameters()).dtype
    decoded = vae_model.decode(latent.to(dtype=vae_dtype))
    array = (
        (decoded.float() * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    ).astype(np.uint8)
    image = Image.fromarray(array)
    bbox = image.getbbox()
    if bbox is not None:
        image = image.crop(bbox)
    return image
