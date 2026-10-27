# OralGPT-CMF

## Overview

**OralGPT-CMF** provides tools for multimodal oral and maxillofacial clinical question answering: clinical-data preparation, Qwen3.5-9B LoRA fine-tuning, multi-turn answer generation, and reference-based AI scoring.

The workflow covers three components:

- **Data preparation** organizes patient records and clinical images into question-answer pairs and supervised fine-tuning datasets.
- **Training and inference** fine-tunes Qwen3.5-9B and evaluates local adapters or OpenAI-compatible API models through sequential patient conversations.
- **Evaluation** scores perception, diagnosis, and treatment answers against reference answers, with question-level records and patient-level summaries.

Training and inference share a conversation format, but not their answer history: training uses reference answers, while inference uses the model's own earlier predictions.

| Directory | Purpose |
| --- | --- |
| `pipeline/` | Clinical-data processing and question-answer construction |
| `sft/data_preparation/` | Patient splits, conversation context, and training samples |
| `sft/configs/`, `sft/train.sh`, `sft/tools/` | Training configurations, launcher, and analysis tools |
| `answer/` | Local and API answer generation, resumable runs, and progress checks |
| `scoring/` | Judge calls, scoring prompts, and metric aggregation |
| `common/`, `prompts/` | Shared conversation history, question routing, and prompts |

## Installation

This subproject currently uses server-specific environments rather than a standalone dependency manifest. The bundled launchers assume the following deployment:

| Component | Expected environment or resource |
| --- | --- |
| API inference and raw-data processing | Conda environment `cmfbench`, initialized through `/root/miniconda3/etc/profile.d/conda.sh` |
| Training and local inference | `/root/autodl-tmp/venvs/qwen35_sft` and `/root/autodl-tmp/LLaMA-Factory-qwen35` |
| Base model | `/root/autodl-tmp/models/Qwen3.5-9B` |
| Background local inference | `screen` |

API inference and scoring require the OpenAI SDK, HTTPX, Pillow, and PyYAML as applicable. Raw-data processing additionally uses OpenPyXL, OpenCV, NumPy, and a MediaPipe version with FaceMesh support. Training and local inference require a LLaMA-Factory installation compatible with Qwen3.5 and the `qwen3_5_nothink` template.

On another host, update the environment, model, dataset, and output paths in the launchers and training YAML files before running them. The local launchers enable offline model loading, so download the required weights beforehand.

Run the examples below from this subproject directory:

```bash
cd /root/autodl-tmp/OralGPT/OralGPT-CMF
export PATH=/root/autodl-tmp/venvs/qwen35_sft/bin:$PATH
```

## Configuration

### Model endpoints

Create `.env` in `OralGPT-CMF/`. API answer generation and scoring use OpenAI-compatible endpoints. By default, scoring reads the same API key and base URL as answer generation; select the judge model separately with `--judge-model`.

```dotenv
ANSWER_OPENAI_API_KEY=your_api_key
ANSWER_OPENAI_BASE_URL=https://api.openai.com/v1
ANSWER_OPENAI_MODEL=your_answer_model
```

Use an endpoint and model that support the requested task. Only send clinical data to services you are authorized to use. Keep credentials and patient data out of Git.

### Datasets and checkpoints

Prepare these local inputs before rebuilding the SFT datasets:

| Input | Location relative to `OralGPT-CMF/` |
| --- | --- |
| First VQA source | `VQA140_final.json` |
| Media for the first source | `../OralFlow/validation/` |
| Second VQA source | `VQA387_final.json` |
| Media for the second source | `../OralFlow/SH9HCMFdata/` |

The preparation scripts organize media through symbolic links. Raw data, generated datasets under `sft/data/`, adapters and checkpoints under `sft/models/`, and runtime outputs under `output/` are local resources, not bundled Git assets.

The `qwen35_lora_v1.yaml`, `qwen35_lora_v2.yaml`, and `qwen35_lora_v3.yaml` configurations retain separate training settings. For a new experiment, choose a new `output_dir`; to resume training, set `resume_from_checkpoint` to the intended checkpoint.

## Quick Start

### Prepare the SFT datasets

Skip this step if the prepared datasets already exist. `turn_level.py` deletes and rebuilds `sft/data/turn_level`; `question_rules.py` requires its output directory not to exist.

```bash
python sft/data_preparation/turn_level.py
python sft/data_preparation/question_rules.py \
  --source sft/data/turn_level \
  --output sft/data/question_rules \
  --question-prompts prompts/oral_question_prompts.yaml
```

### Fine-tune Qwen3.5-9B

Pass the v3 configuration explicitly; the training launcher defaults to v1 when no configuration is supplied.

```bash
bash sft/train.sh sft/configs/qwen35_lora_v3.yaml
```

The v3 configuration uses LoRA with the vision tower and multimodal projector frozen. Checkpoints, training logs, and TensorBoard events are written beneath the configured output directory.

### Generate answers with a local adapter

The following command runs the v3 adapter in the background with its question-rules dataset and matching prompts:

```bash
bash answer/start_local.sh \
  qwen3.5-9b-v3 \
  "$PWD/sft/models/qwen3_5_9b_lora_turn_level_v3" \
  4096 1 \
  --dataset "$PWD/sft/data/question_rules/oralcmf_sft_test_rollout.json" \
  --question-prompts "$PWD/sft/data/question_rules/question_prompts.yaml"
bash answer/progress.sh
```

Rerun the same command after an interruption to resume. The final positional argument, `1`, is the shard count; each additional shard loads another model instance. The underlying Python entry point is `answer/generate_local.py`.

### Generate answers through an API

After configuring `.env`, start with one patient:

```bash
bash answer/run_api.sh --limit 1 --num-workers 1
```

Remove `--limit 1` for the full configured dataset and increase `--num-workers` according to endpoint capacity. Override the configured model with `--answer-model` when needed. The default dataset is `sft/data/turn_level/oralcmf_sft_test_rollout.json`; use `--dataset` to select another prepared rollout file.

The underlying Python entry point is `answer/generate_api.py`. Existing answers and caches are reused; `--force` discards saved answers and caches for the selected model.

### Score the answers

Scoring uses `scoring/prompts/oral_scoring.yaml` by default. Select a new `--run-name` whenever answers or scoring rules change.

```bash
python scoring/score_answers.py \
  --answer-model qwen3.5-9b-v3 \
  --judge-model gpt-5-mini \
  --run-name my_score_run \
  --num-workers 3
```

Set `--answer-model` to the model directory used during answer generation. Scoring requires HTTPS by default; add `--allow-http` only when you explicitly accept unencrypted transmission to a trusted endpoint.

To check input paths and question counts without making API requests:

```bash
python scoring/score_answers.py --answer-model qwen3.5-9b-v3 --dry-run
```

The dry run still requires the existing answer files and a readable `.env` with a valid base URL.

The rubric evaluates reference-supported clinical content and related unsupported claims. Omissions, unsupported additions, repetition, and off-topic content that have not already been penalized in the clinical dimensions can incur an additional combined penalty of 0–20 points. Each question retains its clinical score, additional penalty, evidence quotations from both answers, and final score.

## Optional Raw-Data Construction

The `pipeline/` tools construct question-answer data from clinical records and images under `../OralFlow/SH9HCMFdata/`. This is separate from the SFT preparation above, which reads the two VQA JSON files rather than the pipeline's merged output.

Set `OPENAI_API_KEY` in the shell environment and review the API endpoints in the stage wrappers before running them. Some wrappers default to HTTP services. Panoramic X-ray report generation additionally requires the configured local imaging-agent endpoint; its service launcher is not provided here.

Stage 0 creates eye-masked images under `FP_mosaic`, but failed face detection can leave images unmasked; review the results before sharing them. Stage 1–4 convenience wrappers overwrite prior outputs by default, so do not rerun them over results you need to preserve.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cmfbench
bash pipeline/run_stage0.sh
bash pipeline/run_stage1.sh
bash pipeline/run_stage2.sh
bash pipeline/run_xray_report.sh
bash pipeline/run_stage3.sh
bash pipeline/run_stage4.sh
python pipeline/build_dataset.py \
  --input-dir output/pipeline \
  --output-file output/pipeline/cmf_sft_dataset.json
```

## Outputs

| Artifact | Location |
| --- | --- |
| Constructed clinical data | `output/pipeline/` |
| Prepared SFT datasets | `sft/data/turn_level/`, `sft/data/question_rules/` |
| Adapters and checkpoints | `sft/models/`, unless overridden in the training configuration |
| Model answers | `output/<group>/<patient>/evaluation/<model>/answers.json` |
| Question-level scores and patient summaries | `output/<group>/<patient>/evaluation/<model>/scoring/<judge>/<run>/` |
| Batch scoring summary | `output/scoring/<model>/<judge>/<run>/summary.json` |
| Answer-launcher logs | `output/logs/` |

Direct Python entry points print progress to standard output. Training writes `train.log` and TensorBoard events beneath its configured output directory. Credentials, clinical data, model artifacts, caches, and generated results must remain outside Git.
