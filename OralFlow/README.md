<p align="center">
  <img src="assets/oralflow_logo.png" alt="OralFlow-Bench logo" width="200">
</p>

<h1 align="center">OralFlow-Bench</h1>

## 🔍 Overview

**OralFlow-Bench** is a benchmark and evaluation framework for studying multimodal, longitudinal memory in oral and maxillofacial clinical reasoning. Unlike static medical visual question answering, it releases clinical information progressively: a model must retain relevant findings, retrieve earlier evidence, reconcile later observations with its existing state, and reason across modalities or timepoints.

OralFlow-Bench provides two complementary tracks:

- **OralFlow-Literature** transforms open-access case-report PDFs into time-indexed `perception → treatment → followup` trajectories. MinerU parses text, tables, figures, and captions, while an extraction model and verifier iteratively construct and validate the clinical timeline.
- **OralFlow-Cohort** aligns multi-turn patient records with clinical images and organizes them into ordered stages, including patient profile, facial photographs, intraoral photographs, radiographic data, CT, and TMJ examinations.

Both tracks share the same core workflow: trajectory construction, atomic-evidence extraction, evidence-graph construction, task generation and validation, evidence attribution, rubric generation, and sequential memory evaluation.

## ⚙️ Installation

```bash
conda create -n cmfbench python=3.10 -y
conda activate cmfbench
pip install -r requirement.txt
playwright install chromium
```

## 🔑 Configuration

Create `.env` in the repository root. All model endpoints use the OpenAI-compatible API format. Role-specific settings fall back to `OPENAI_*` when omitted.

```dotenv
# Shared fallback
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your_default_model

# Benchmark construction
BENCHMARK_OPENAI_API_KEY=your_api_key
BENCHMARK_OPENAI_BASE_URL=https://api.openai.com/v1
BENCHMARK_OPENAI_MODEL=your_generation_model

# Model under evaluation
ANSWER_OPENAI_API_KEY=your_api_key
ANSWER_OPENAI_BASE_URL=https://api.openai.com/v1
ANSWER_OPENAI_MODEL=your_answer_model

# Question validation and scoring
VERIFIER_OPENAI_API_KEY=your_api_key
VERIFIER_OPENAI_BASE_URL=https://api.openai.com/v1
VERIFIER_OPENAI_MODEL=your_verifier_model
```

Configure only the services required by the selected memory methods:

```dotenv
# SummaryMemory and LangMem
MEMO_OPENAI_API_KEY=your_api_key
MEMO_OPENAI_BASE_URL=http://127.0.0.1:8004/v1
MEMO_OPENAI_MODEL=qwen2.5-7b-instruct-memory

# VectorMemory, Mem0, LangMem, and Graphiti
EMBEDDING_OPENAI_API_KEY=EMPTY
EMBEDDING_OPENAI_BASE_URL=http://127.0.0.1:8005/v1
EMBEDDING_MODEL=qwen3-embedding-0.6b

# Mem0
MEM0_OPENAI_API_KEY=your_api_key
MEM0_OPENAI_BASE_URL=https://api.openai.com/v1
MEM0_OPENAI_MODEL=your_mem0_model

# Graphiti
GRAPHITI_OPENAI_API_KEY=your_api_key
GRAPHITI_OPENAI_BASE_URL=https://api.openai.com/v1
GRAPHITI_OPENAI_MODEL=your_graphiti_model
GRAPHITI_NEO4J_URI=bolt://127.0.0.1:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=your_neo4j_password
```

### Service requirements

| Stage or method | Required service |
| --- | --- |
| OralFlow-Cohort trajectory/evidence construction | Benchmark LLM |
| Task generation and validation | Benchmark LLM + verifier LLM |
| OralFlow-Literature ingestion and timeline construction | MinerU + benchmark LLM + verifier LLM |
| Model-perception trajectory | Multimodal answer model |
| Perception and answer scoring | Verifier LLM |
| `single_stage_memory`, `full_context_memory` | No additional memory service |
| `summary_memory` | Memo LLM |
| `vector_memory` | Embedding model |
| `mem0_memory` | Mem0 LLM + embedding model |
| `langmem_memory` | Memo LLM + embedding model |
| `graphiti_memory` | Graphiti LLM + embedding model + Neo4j |

## 🚀 Quick Start

Every batch command requires exactly one of `--all` or `--limit N`. Use `--limit 1` for test, then switch to `--all` for a complete run. Completed cases are skipped automatically; `--force` reruns the selected stage.

### OralFlow-Cohort

Place the LLaMA-Factory-style patient dataset at `oralgpt_cmf_llamafactory_sft_dataset.json` and the referenced clinical media under `SH9HCMFdata/`.

```bash
# Build trajectories, evidence, and evidence graphs
bash scripts/run_step1_step2.sh --limit 1 --num-workers 1 --stage-workers 2

# Generate and validate benchmark tasks
bash scripts/run_step3.sh --limit 1 --num-workers 1 --task-workers 4

# Freeze model answers
bash scripts/run_step4_answers.sh --limit 1 --num-workers 1 \
  --trajectories standard_trajectory \
  --methods full_context_memory \
  --answer-workers 1 --method-workers 1

# Score the frozen answers
bash scripts/run_step4_scoring.sh --limit 1 --num-workers 1 \
  --trajectories standard_trajectory \
  --methods full_context_memory \
  --score-workers 1 --method-workers 1
```

To compare multiple conditions and memory systems:

```bash
bash scripts/run_step4_answers.sh --all --num-workers 1 \
  --trajectories standard_trajectory,short_noisy,no_ct \
  --methods single_stage_memory,full_context_memory,summary_memory,vector_memory \
  --answer-workers 2 --method-workers 1

bash scripts/run_step4_scoring.sh --all --num-workers 1 \
  --trajectories standard_trajectory,short_noisy,no_ct \
  --methods single_stage_memory,full_context_memory,summary_memory,vector_memory \
  --score-workers 1 --method-workers 1
```

### Model-perception evaluation

Use the same VLM to generate visual observations and answer downstream benchmark questions.

```bash
bash scripts/run_perception_trajectory.sh --limit 1 --num-workers 1 \
  --question-workers 1 --model medgemma \
  --base-url http://127.0.0.1:8002/v1

bash scripts/run_perception_evaluation.sh --limit 1 --num-workers 1 \
  --question-workers 1 --model medgemma

bash scripts/run_step4_answers.sh --limit 1 --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods full_context_memory \
  --answer-model medgemma \
  --answer-base-url http://127.0.0.1:8002/v1 \
  --answer-workers 1 --method-workers 1

bash scripts/run_step4_scoring.sh --limit 1 --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods full_context_memory \
  --answer-model medgemma \
  --answer-base-url http://127.0.0.1:8002/v1 \
  --score-workers 1 --method-workers 1
```

### OralFlow-Literature

Place source PDFs under `reports/pdf/` before running the report pipeline.

```bash
bash scripts/run_step0_step1_report.sh --limit 1 --num-workers 1
bash scripts/run_step2_report.sh --limit 1 --num-workers 1 --stage-workers 2
bash scripts/run_step3_report.sh --limit 1 --num-workers 1 --task-workers 4

bash scripts/run_step4_report_answers.sh --limit 1 --num-workers 1 \
  --methods full_context_memory \
  --answer-workers 1 --method-workers 1

bash scripts/run_step4_report_scoring.sh --limit 1 --num-workers 1 \
  --methods full_context_memory \
  --score-workers 1 --method-workers 1
```

## 📚 Reference

```bibtex
@inproceedings{anonymous2027oralflow,
  title     = {{OralFlow}: A Longitudinal Multimodal Benchmark across Clinical Workflows in Dentistry},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2027},
  note      = {Under review}
}
```
