#!/usr/bin/env python3
"""Merge *_video.m4s and *_audio.m4s pairs into MP4 files using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

VIDEO_BILIBILI_ROOT = Path(__file__).resolve().parent / "bilibili_videos"


def find_pairs(folder: Path) -> Iterable[Tuple[Path, Path, Path]]:
    """Yield (video_path, audio_path, output_path) tuples for every usable pair."""
    for video_path in sorted(folder.glob("*_video.m4s")):
        stem = video_path.name[:-len("_video.m4s")]
        audio_path = folder / f"{stem}_audio.m4s"
        output_path = folder / f"{stem}.mp4"
        if not audio_path.exists():
            print(f"⚠️  缺少音频文件，跳过: {audio_path.name}")
            continue
        yield video_path, audio_path, output_path


def run_ffmpeg(video_path: Path, audio_path: Path, output_path: Path, overwrite: bool) -> bool:
    """Run ffmpeg to merge a single audio/video pair."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        str(output_path),
    ]
    if overwrite:
        cmd.insert(-1, "-y")
    else:
        cmd.insert(-1, "-n")

    print(f"➡️  合并: {video_path.name} + {audio_path.name} -> {output_path.name}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 完成: {output_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ ffmpeg 合并失败: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="批量合并m4s音视频为mp4")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(VIDEO_BILIBILI_ROOT),
        help=f"包含m4s文件的目录 (默认: {VIDEO_BILIBILI_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标mp4已存在则覆盖",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"❌ 目录不存在: {folder}")
        return 1

    pairs = list(find_pairs(folder))
    if not pairs:
        print("⚠️  未找到可合并的m4s文件对")
        return 0

    print(f"在 {folder} 中发现 {len(pairs)} 组待合并文件")
    success = 0
    for video_path, audio_path, output_path in pairs:
        if output_path.exists() and not args.overwrite:
            print(f"⏭️  已存在MP4且未指定 --overwrite，跳过: {output_path.name}")
            continue
        if run_ffmpeg(video_path, audio_path, output_path, args.overwrite):
            success += 1

    print(f"\n总结: 成功合并 {success}/{len(pairs)} 组")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
