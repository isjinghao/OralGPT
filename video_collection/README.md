# Video Collection Tools

Scripts for collecting oral health education videos from YouTube and Bilibili.

## Scripts

| File | Description |
|---|---|
| `youtube_scraping.py` | YouTube video scraper with AI-powered keyword generation and relevance filtering |
| `bilibili_scraping.py` | Bilibili video scraper with AI-powered keyword generation and relevance filtering |
| `merge_m4s.py` | Utility to merge Bilibili `.m4s` audio/video segments into `.mp4` files via ffmpeg |

## Dependencies

```bash
pip install requests beautifulsoup4 yt-dlp
# Optional: pip install selenium  (for YouTube search fallback)
```

`merge_m4s.py` requires [ffmpeg](https://ffmpeg.org/) installed on system PATH.

## Usage

### YouTube

```bash
python youtube_scraping.py \
    --category cinematic_arts \
    --max-videos 100 \
    --api-key <OPENAI_COMPATIBLE_API_KEY> \
    --base-url <API_BASE_URL>
```

### Bilibili

```bash
python bilibili_scraping.py \
    --category cinematic_arts \
    --max-videos 100 \
    --api-key <OPENAI_COMPATIBLE_API_KEY> \
    --base-url <API_BASE_URL>
```

### Merge Bilibili m4s

```bash
python merge_m4s.py <folder_with_m4s_files> [--overwrite]
```

## Configuration

Output paths default to `./youtube_videos`, `./bilibili_videos`, and `./tracking` relative to the script location. Override via environment variables:

```bash
export VIDEO_YOUTUBE_ROOT=/path/to/youtube_videos
export VIDEO_BILIBILI_ROOT=/path/to/bilibili_videos
export TRACKING_ROOT=/path/to/tracking
```

## Supported Categories

- `cinematic_arts` -- film analysis, cinematography, editing techniques
- `static_visual_arts` -- painting, photography, visual design
- `stage_performing_arts` -- theater, dance, live performance
- `game_arts` -- game design, visual effects, game cinematics
