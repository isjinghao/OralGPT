# OralGPT-X

Training data preparation scripts and **OralGPT-X-Bench** evaluation for BAGEL-based oral image generation.

## Layout

```text
OralGPT-X/
├── scripts/
│   ├── cbct_dataset/
│   ├── mri_dataset/
│   └── orthosurgery_dataset/
└── oralgpt_x_bench/          # BAGEL evaluation plug-in (push to GitHub)
    ├── scripts/run_cbct.sh
    ├── infer/
    ├── metrics/
    └── summarize/
```

## OralGPT-X-Bench

Independent benchmark evaluation for BAGEL `unified_edit` / `t2i`. Does **not** modify Bagel source.

| Benchmark | Task |
|-----------|------|
| **cbct** | Low-dose → standard CBCT restoration |
| **ortho** | Pre→post orthognathic simulation |
| **mri** | T1↔T2 modality translation |
| **t2i** | Caption → radiograph (planned) |

See [oralgpt_x_bench/README.md](./oralgpt_x_bench/README.md).

### Quick start (CBCT)

```bash
export BAGEL_ROOT=/path/to/Bagel
export model_path=/path/to/checkpoint
export output_path=/path/to/eval_output
bash oralgpt_x_bench/scripts/run_cbct.sh
```
