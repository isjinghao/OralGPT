#!/usr/bin/env python3
"""Distributed unified_edit inference for OralGPT-X-Bench."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from path_utils import bench_root, ensure_bagel_on_path, resolve_path

BENCH_ROOT = bench_root()
ensure_bagel_on_path()

from data.data_utils import pil_img2rgb  # noqa: E402
from infer.bagel_loader import load_bagel_for_edit  # noqa: E402
from infer.edit_inference import run_edit_inference  # noqa: E402


def setup_distributed() -> None:
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_samples(metadata_file: Path) -> list[dict]:
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported metadata format: {metadata_file}")


def load_source_image(bench_data_root: Path, sample: dict) -> Image.Image:
    rel = sample["source"]["image_path"]
    return pil_img2rgb(Image.open(bench_data_root / rel))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_file", type=Path, required=True)
    parser.add_argument("--bench_data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--bagel-root", type=Path, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bagel_root is not None:
        ensure_bagel_on_path(args.bagel_root)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"

    metadata_file = resolve_path(args.metadata_file, BENCH_ROOT)
    bench_data_root = resolve_path(args.bench_data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed + rank)
    samples = load_samples(metadata_file)
    chunk = (len(samples) + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, len(samples))
    if rank == 0:
        print(f"Loaded {len(samples)} samples from {metadata_file}")
        print(f"Saving predictions to {output_dir}")

    runtime = load_bagel_for_edit(args.model_path, device=device, max_latent_size=args.max_latent_size)

    for idx in range(start, end):
        sample = samples[idx]
        sample_id = sample["id"]
        task_type = sample.get("task_type", "default")
        out_dir = output_dir / task_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample_id}.png"
        if out_path.exists():
            print(f"[rank {rank}] skip existing {out_path}")
            continue

        source = load_source_image(bench_data_root, sample)
        prompt = sample["instruction"]
        print(f"[rank {rank}] {idx - start + 1}/{end - start} {sample_id}")

        pred = run_edit_inference(
            runtime,
            source,
            prompt,
            num_timesteps=args.num_timesteps,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            timestep_shift=args.timestep_shift,
        )
        pred.save(out_path)

    dist.barrier()
    if rank == 0:
        print("Inference complete.")


if __name__ == "__main__":
    main()
