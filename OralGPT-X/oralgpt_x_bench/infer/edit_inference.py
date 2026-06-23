"""Image editing inference loop (adapted from Bagel eval/gen/gen_images_mp_kris.py)."""

from __future__ import annotations

import copy

import numpy as np
import torch
from PIL import Image

from modeling.bagel.qwen2_navit import NaiveCache

from .bagel_loader import BagelEditRuntime


def _apply_scale(width: int, height: int, scale: float) -> tuple[int, int]:
    def _make_divisible(value: float, stride: int = 16) -> int:
        return max(stride, int(round(value / stride) * stride))

    new_width = _make_divisible(width * scale)
    new_height = _make_divisible(height * scale)
    return new_width, new_height


def _move_to_device(generation_input: dict, device: str) -> dict:
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            generation_input[key] = value.to(device)
    return generation_input


def run_edit_inference(
    runtime: BagelEditRuntime,
    source_image: Image.Image,
    prompt: str,
    *,
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 1.5,
    cfg_interval: list[float] | None = None,
    cfg_renorm_min: float = 0.0,
    timestep_shift: float = 3.0,
    max_image_size: int = 1024,
    min_image_size: int = 512,
) -> Image.Image:
    if cfg_interval is None:
        cfg_interval = [0.0, 1.0]

    gen_model = runtime.model
    vae_model = runtime.vae_model
    vae_transform = runtime.vae_transform
    vit_transform = runtime.vit_transform
    tokenizer = runtime.tokenizer
    new_token_ids = runtime.new_token_ids
    device = runtime.device

    w, h = source_image.size
    scale = min(max_image_size / max(w, h), 1.0)
    scale = max(scale, min_image_size / min(w, h))
    w, h = _apply_scale(w, h, scale)
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = _apply_scale(w, h, scale)

    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens, new_rope = [0], [0]

    for image in [source_image]:
        generation_input, newlens, new_rope = gen_model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=vae_transform,
            new_token_ids=new_token_ids,
        )
        generation_input = _move_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vae(
                vae_model, past_key_values, **generation_input
            )

        generation_input, newlens, new_rope = gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        generation_input = _move_to_device(generation_input, device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_vit(past_key_values, **generation_input)

    cfg_text_past_key_values = copy.deepcopy(past_key_values)
    generation_input_cfg_text = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(h, w)],
    )
    generation_input_cfg_text = _move_to_device(generation_input_cfg_text, device)

    cfg_img_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0]
    cfg_img_new_rope = [0]
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input_cfg_img = _move_to_device(generation_input_cfg_img, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        cfg_img_past_key_values = gen_model.forward_cache_update_text(
            cfg_img_past_key_values, **generation_input_cfg_img
        )
    generation_input_cfg_img = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        image_sizes=[(h, w)],
    )
    generation_input_cfg_img = _move_to_device(generation_input_cfg_img, device)

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
        image_sizes=[(h, w)],
        new_token_ids=new_token_ids,
    )
    generation_input = _move_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = gen_model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type="serial_text_img",
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type="text_channel",
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )

    latent = unpacked_latent[0]
    latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h // 8, w // 8)
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
