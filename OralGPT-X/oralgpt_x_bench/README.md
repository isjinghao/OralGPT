# OralGPT-X-Bench

OralGPT-X-Bench 是面向 [BAGEL](https://github.com/ByteDance-Seed/Bagel) 的**独立评估插件**，用于口腔/颌面医学图像 **Edit** 与 **T2I** 任务。  
**不修改 BAGEL 官方代码**；仅在运行时把 `BAGEL_ROOT` 加入 `sys.path` 并 import `modeling.*` / `data.*`。

---

## Benchmark 一览

| Benchmark | 任务 | Stage 1 脚本 | Stage 2 指标 | 主指标 | 分组维度 |
|-----------|------|-------------|-------------|--------|----------|
| **cbct** | 低剂量 → 标准剂量 CBCT 复原 | `gen_edit_mp.py` | `metrics/pixel.py` | SSIM, PSNR, NMI | `task_type`, `volume_id` |
| **ortho** | 正颌术前 → 术后 X 光模拟 | `gen_edit_mp.py` | `metrics/pixel.py` | SSIM, LPIPS | `modality`, `batch` |
| **mri** | T1 ↔ T2 MRI 模态转换 | `gen_edit_mp.py` | `metrics/pixel.py` | SSIM, NMI | `task_type`, `cohort` |
| **t2i** | 文本 → 口内/口腔影像 | `gen_t2i_mp.py` | `metrics/judge_t2i.py` | Consistency, Clinical_plausibility | `category` |

**Edit 共用像素指标：** SSIM, PSNR, NMI, MAE, LPIPS（需 `pip install lpips`，否则 LPIPS 为 null）。  
**T2I GPT 指标：** Consistency, Realism, Clinical_plausibility（0–2 分，需 `OPENAI_API_KEY`）。

---

## 评估流程（三阶段）

与 BAGEL 官方 `run_kris.sh` / `run_wise.sh` 一致：

```text
scripts/run_{cbct,ortho,mri,t2i}.sh
    │
    ├─ Stage 1  Inference
    │     Edit: infer/gen_edit_mp.py   (source + instruction → pred PNG)
    │     T2I:  infer/gen_t2i_mp.py   (prompt → pred PNG)
    │     输出 → $output_path/inference/{benchmark}/
    │
    ├─ Stage 2  Metrics
    │     Edit: metrics/pixel.py      (pred vs GT，写 JSONL)
    │     T2I:  metrics/judge_t2i.py  (GPT-4o 评 pred，写 JSONL)
    │
    └─ Stage 3  Summarize
          summarize/summarize.py → {benchmark}_summary.json + leaderboard.csv
```

---

## 环境准备

```bash
conda activate bagel   # 或你的 BAGEL 训练/推理环境

cd /path/to/Bagel
pip install -r requirements.txt
# huggingface-hub 需与 BAGEL 一致，例如 0.29.1

pip install -r /path/to/OralGPT/OralGPT-X/oralgpt_x_bench/requirements-extra.txt
# scikit-image, lpips, pyarrow, tqdm, ...
pip install lpips   # 可选但推荐
```

### 必需环境变量

```bash
export BAGEL_ROOT=/path/to/Bagel
export model_path=/path/to/checkpoint    # 含 ema.safetensors、llm_config.json 等
export output_path=/path/to/eval_results
export GPUS=1                              # 单机 smoke test 建议 1；全量可 8
```

### 可选环境变量

| 变量 | 说明 |
|------|------|
| `FULL_EVAL=1` | 使用全量 test metadata + 从 parquet export（见下文） |
| `parquet_root` | 全量 parquet 根目录（各 benchmark 默认值见「数据来源」） |
| `bench_data_root` | PNG 数据根目录；默认 `benchmark/{name}/examples` |
| `T2I_RESOLUTION` | T2I 生成分辨率，默认 `512` |
| `OPENAI_API_KEY` | T2I GPT judge 必需（无则 `JUDGE_MODE=auto` 走 stub） |
| `OPENAI_BASE_URL` | 可选，兼容 OpenAI 兼容 API |
| `JUDGE_MODEL` | 默认 `gpt-4o` |
| `JUDGE_MODE` | `auto` / `openai` / `stub` |

---

## 快速开始（Smoke Test，各 5 条 example）

仓库内已含 **cbct / ortho / mri / t2i 各 5 条** 公开样例（PNG + `metadata.examples.json`），无需 parquet 即可跑通流程。

```bash
export BAGEL_ROOT=/path/to/Bagel
export model_path=/data/OralGPT/OralGPT-X/models/BAGEL-7B-MoT
export output_path=/tmp/oralgpt_x_bench_smoke
export GPUS=1

BENCH=/path/to/OralGPT/OralGPT-X/oralgpt_x_bench

bash $BENCH/scripts/run_cbct.sh
bash $BENCH/scripts/run_ortho.sh
bash $BENCH/scripts/run_mri.sh
bash $BENCH/scripts/run_t2i.sh
```

单卡 A100 80GB + bf16 加载约 27GB 显存；Edit 小图 ~10–20s/条，Ortho 大图 ~30–60s/条，T2I 512² ~12s/条。

---

## 各 Benchmark 运行命令

### CBCT

```bash
# 默认 5 examples
bash oralgpt_x_bench/scripts/run_cbct.sh

# 全量 test（1530 slices，3 个剂量任务）
export FULL_EVAL=1
export parquet_root=/data/OralGPT/OralGPT-X/dataset_CBCT_Low-Dose_to_Standard/test
bash oralgpt_x_bench/scripts/run_cbct.sh
```

### Ortho

```bash
bash oralgpt_x_bench/scripts/run_ortho.sh

export FULL_EVAL=1
export parquet_root=/data/OralGPT/OralGPT-X/dataset_OrthoSurgery/test
bash oralgpt_x_bench/scripts/run_ortho.sh
```

### MRI

```bash
bash oralgpt_x_bench/scripts/run_mri.sh

export FULL_EVAL=1
export parquet_root=/data/OralGPT/OralGPT-X/dataset_MRI_T1_T2/test
bash oralgpt_x_bench/scripts/run_mri.sh
```

### T2I

```bash
# 推理 + stub judge（无 API key 时）
bash oralgpt_x_bench/scripts/run_t2i.sh

# 真实 GPT 评分
export OPENAI_API_KEY=sk-...
export JUDGE_MODE=openai
rm -f $output_path/metrics/judge_t2i.jsonl   # 需重评时删除
bash oralgpt_x_bench/scripts/run_t2i.sh
```

---

## 输出目录结构

```text
$output_path/
├── inference/
│   ├── cbct/{task_type}/{sample_id}.png
│   ├── ortho/{task_type}/{sample_id}.png
│   ├── mri/{task_type}/{sample_id}.png
│   └── t2i/{sample_id}.png
├── metrics/
│   ├── pixel_cbct.jsonl
│   ├── pixel_ortho.jsonl
│   ├── pixel_mri.jsonl
│   └── judge_t2i.jsonl
└── summary/
    ├── cbct_summary.json
    ├── ortho_summary.json
    ├── mri_summary.json
    ├── t2i_summary.json
    └── leaderboard.csv
```

`{benchmark}_summary.json` 含 `overall`（mean/min/max）及 `by_group` 分组统计。

---

## Metadata 与数据格式

### 通用约定

- **Smoke test：** `benchmark/{name}/metadata.examples.json` + `benchmark/{name}/examples/`（进 git）
- **全量评估：** `benchmark/{name}/metadata.test.json` + `benchmark_data/{name}/`（不进 git，本地 export）
- Edit 类 metadata 为 **JSON**，顶层含 `samples` 数组；T2I 也支持 **JSONL**（每行一条）
- 图像路径均相对于 `bench_data_root`

### Edit 类（cbct / ortho / mri）— 单条 sample  schema

```json
{
  "id": "unique_sample_id",
  "benchmark": "cbct",
  "task_family": "edit_restoration",
  "task_type": "cbct_78_to_333",
  "split": "test",
  "source": {
    "image_path": "source/cbct_78_to_333/xxx.png"
  },
  "target": {
    "image_path": "target/cbct_78_to_333/xxx.png",
    "role": "pixel_gt"
  },
  "instruction": "Task: ... Instruction: ...",
  "metadata": {
    "volume_id": "127_350257",
    "modality": "pan",
    "batch": "0526配对三片",
    "cohort": "Guizhou"
  }
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `id` | ✅ | 唯一 ID；预测文件名 `{id}.png` |
| `task_type` | ✅ | 子任务目录名，如 `cbct_78_to_333`、`orthosurgery_pan_pre_to_post`、`mri_t1_to_t2` |
| `source.image_path` | ✅ | 相对 `bench_data_root` 的输入图 |
| `target.image_path` | ✅ | GT 图（metric 基准分辨率） |
| `instruction` | ✅ | 传给 BAGEL unified_edit 的文本 |
| `metadata.*` | 推荐 | 用于 summarize 分组（见 registry） |

**推理行为（Edit）：** 默认 `min_image_size=512`，小图会被放大后再生成；输出 pred 通常为 512² 或按长边缩放。  
**Metric 行为：** 将 **pred 下采样到 GT 尺寸** 再算 SSIM/PSNR 等（GT 保持原分辨率）。

### T2I — 单条 sample schema

```json
{
  "id": "intraoral_caries_001",
  "benchmark": "t2i",
  "task_family": "text_to_image",
  "category": "pathology_caries",
  "split": "test",
  "prompt": "Please generate an intraoral image showing dental caries.",
  "metadata": {
    "modality": "intraoral_photo",
    "finding": "caries",
    "language": "en"
  }
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `id` | ✅ | 预测 `{id}.png` |
| `prompt` | ✅ | T2I 生成文本 |
| `category` | 推荐 | summarize 按 category 分组 |

当前 5 条 smoke prompt 见 `benchmark/t2i/metadata.examples.json`（caries / healthy / gingivitis / missing tooth / crown）。

---

## 从 Parquet 导出全量 Benchmark 数据

各 `tools/export_*_benchmark.py` 从 BAGEL 训练 parquet 导出 PNG + `metadata.test.json`：

```bash
cd oralgpt_x_bench
export PYTHONPATH=.

# CBCT
python tools/export_cbct_benchmark.py \
  --parquet-root /data/OralGPT/OralGPT-X/dataset_CBCT_Low-Dose_to_Standard/test \
  --bench-data-root /path/to/benchmark_data/cbct \
  --output-metadata benchmark/cbct/metadata.test.json

# Ortho
python tools/export_ortho_benchmark.py \
  --parquet-root /data/OralGPT/OralGPT-X/dataset_OrthoSurgery/test \
  --bench-data-root /path/to/benchmark_data/ortho \
  --output-metadata benchmark/ortho/metadata.test.json

# MRI
python tools/export_mri_benchmark.py \
  --parquet-root /data/OralGPT/OralGPT-X/dataset_MRI_T1_T2/test \
  --bench-data-root /path/to/benchmark_data/mri \
  --output-metadata benchmark/mri/metadata.test.json \
  --max-per-task 3 --max-total 5   # 仅 smoke 时可限条数
```

Parquet 行需含：`image_list`（[source_bytes, target_bytes]）、`instruction_list`、`pair_id` 等（与训练数据一致）。

---

## 指标说明

### Edit — `metrics/pixel.py`

| 指标 | 方向 | 说明 |
|------|------|------|
| SSIM | ↑ | 结构相似度 [0,1] |
| PSNR | ↑ | 峰值信噪比 (dB) |
| NMI | ↑ | 归一化互信息 |
| MAE | ↓ | 灰度 MAE（0–255） |
| LPIPS | ↓ | AlexNet 感知距离（需 lpips） |

### T2I — `metrics/judge_t2i.py`（WISE 风格简化）

| 指标 | 方向 | 说明 |
|------|------|------|
| Consistency | ↑ (0–2) | 与 prompt 是否一致 |
| Realism | ↑ (0–2) | 是否像真实临床照片 |
| Clinical_plausibility | ↑ (0–2) | 口腔解剖/病灶是否合理 |

---

## 目录结构

```text
oralgpt_x_bench/
├── README.md
├── DESIGN.md
├── requirements-extra.txt
├── config/benchmarks.yaml
├── benchmark/
│   ├── cbct/metadata.examples.json + examples/
│   ├── ortho/metadata.examples.json + examples/
│   ├── mri/metadata.examples.json + examples/
│   └── t2i/metadata.examples.json
├── scripts/
│   ├── env.sh
│   ├── run_cbct.sh | run_ortho.sh | run_mri.sh | run_t2i.sh
├── tools/
│   ├── export_cbct_benchmark.py
│   ├── export_ortho_benchmark.py
│   └── export_mri_benchmark.py
├── infer/
│   ├── bagel_loader.py      # bf16 加载 BAGEL
│   ├── edit_inference.py
│   ├── gen_edit_mp.py
│   ├── t2i_inference.py
│   └── gen_t2i_mp.py
├── metrics/
│   ├── pixel.py
│   └── judge_t2i.py
└── summarize/
    ├── registry.py
    └── summarize.py
```

---

## 数据来源与规模

| 数据集 | Parquet test 路径 | Test 规模 | HF |
|--------|-------------------|-----------|-----|
| CBCT | `dataset_CBCT_Low-Dose_to_Standard/test/` | 1530（3 任务 × 510） | `OralGPT/OralGPT-X` |
| Ortho | `dataset_OrthoSurgery/test/` | 147（49 cases × pan/la/xf） | 同上 |
| MRI | `dataset_MRI_T1_T2/test/` | 1212（t1→t2 + t2→t1） | 同上 |
| T2I | 无固定 parquet | 自建 prompt 列表 | — |

---

## 部署到另一台机器

1. 拷贝 `OralGPT/OralGPT-X/oralgpt_x_bench/`（含各 benchmark 的 5 条 examples）
2. 安装 BAGEL 代码 + checkpoint + conda 环境
3. 设置 `BAGEL_ROOT`、`model_path`、`output_path`、`GPUS`
4. 先跑默认 `run_*.sh` 验证流程
5. 全量评估：挂载 parquet → `FULL_EVAL=1` 或手动 export → 再跑 `run_*.sh`
6. T2I 打分：配置 `OPENAI_API_KEY`

**常见问题：**

- `huggingface-hub` 版本与 BAGEL 冲突 → 使用 `huggingface_hub==0.29.1`
- GPU 被占满 → 释放其他进程；Edit 已默认 bf16 加载 (~27GB)
- T2I judge 全 null → 未设 API key，使用 stub 模式

---

## 相关文档

- 架构概要：`DESIGN.md`
- BAGEL 官方评估：`BAGEL/EVAL.md`（KRIS/WISE/GEdit 等 GPT bench 参考）
- 任务设计（若存在）：`OralGPT-X/OralGPT-X_Benchmark_Design.md`
