# OralGPT-Plus

Training code and data-curation tools for **OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis** (CVPR 2026).

## Repository Structure

```
OralGPT-Plus/
├── training/
│   ├── mini-o3-rl.tar.gz        # Packaged RL training source tree
│   └── README.md                 # Original training README
└── tools/
    ├── data_process_for_*.py     # Raw dataset processors (4 sources)
    ├── step_0_data*_proposal.py  # DentalProbe Step 0 -- region proposal generation
    ├── step_1_*_filtering.py     # DentalProbe Step 1 -- multi-agent VLM filtering
    ├── step_2_*_rewr*.py         # DentalProbe Step 2 -- GPT-based dialogue rewriting
    ├── yolo2coco.py              # YOLO-to-COCO format conversion
    ├── infer_dino.py             # DINO detection for tooth-ID assignment
    ├── llm_judge.py              # LLM-as-a-judge evaluation scorer
    ├── filter_by_keywords.py     # Keyword-based sample filtering
    ├── visual_for_Tufts_Dental_Database.py
    └── draw_bar_chart_for_human_scoring_DentalProbe_data_quality.py
```

---

## 1. RL Training (Mini-o3 Based)

The RL training code is built on top of [Mini-o3](https://github.com/Mini-o3/Mini-o3) (a VeRL-based multi-turn agentic RL framework) and extends it with a new visual tool, `mirror_grounding`, for panoramic X-ray analysis.

### 1.1 Setup

```bash
cd training
tar xzf mini-o3-rl.tar.gz
cd mini-o3
pip install -r requirements.txt
pip install -e .
```

### 1.2 Main Modifications over Upstream Mini-o3

- Added a `<mirror_grounding>` visual tool. Async and SPMD vLLM rollouts route these calls through `mirror_image`, with tool-trigger bookkeeping kept consistent with the existing crop flow.
- Reward scoring (`verl/utils/reward_score/general_qa_tool.py`, `general_qa_tool_mc.py`) judges `<mirror_grounding>` steps the same way as `<grounding>` for format and accuracy rewards.
- Training analytics treat `crop_mirror` like the crop tool so regex checks, statistics, and reward reporting stay accurate.
- Separate wandb counters log `<grounding>` and `<mirror_grounding>` usage in `verl/trainer/ppo/ray_trainer.py`.

See `tool_modification_summary.md` inside the archive for the detailed change log.

### 1.3 Judge API Key

Reward computation can call an external judge model. Set the API key before training or validation:

```bash
export API_KEY="your_judge_api_key"
```

### 1.4 Data Layout

The scripts expect the following directory layout under `BASE_IMAGE_DIR`:

```
BASE_IMAGE_DIR/
├── rl_train/
│   ├── mmoral-resoning-train-1k.json   # 1k training set (used by mirror_train)
│   └── train.json                       # Full training set (used by val scripts)
├── rl_val_easy/
│   └── val.json
├── rl_val_medium/
│   └── val.json
├── rl_val_hard/
│   └── val.json
└── rl_val_opg/
    └── mmoral-reasoning_opg.json        # OPG benchmark split
```

Each JSON file is a list of samples with `images` (image paths) and `solution` (ground-truth answer) fields.

### 1.5 Training Commands

All training is launched via `verl.trainer.main_ppo` with Hydra-style config overrides. The two main entry scripts differ only in model size and batch parameters.

#### 7B Model Training

```bash
export BASE_IMAGE_DIR="/path/to/your/data"
export API_KEY="your_judge_api_key"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.system_prompt="tool_crop_mirror" \
    data.train_files=[${BASE_IMAGE_DIR}/rl_train/mmoral-resoning-train-1k.json] \
    data.val_files=[${BASE_IMAGE_DIR}/rl_val_easy/val.json,${BASE_IMAGE_DIR}/rl_val_medium/val.json,${BASE_IMAGE_DIR}/rl_val_hard/val.json] \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=0 \
    data.tool_call_penalty=0 \
    data.general_qa_reward_fn="general_qa_tool_mc" \
    data.gpt_general_qa_reward_fn="general_qa_tool_condition_curiosity" \
    data.gpt_extract_answer=True \
    data.extract_answer_tags="strict" \
    data.return_raw_chat=True \
    data.gpt_threads=80 \
    data.tool_call="crop_mirror" \
    data.use_tgt_size=False \
    data.max_pixels=2000000 \
    data.min_pixels=40000 \
    reward_model.reward_manager=naive_multithreads_tool \
    actor_rollout_ref.model.path=/path/to/qwen2_5vl-7b-coldstart-ckpt \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'"'"'image'"'"': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=12 \
    'actor_rollout_ref.rollout.val_limit_mm_per_prompt={'"'"'image'"'"': 12}' \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
    actor_rollout_ref.rollout.vllm_infer_batch_size=4 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MMOral-R1-Mirror' \
    trainer.experiment_name='7b-mirror-train' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.default_local_dir=./save/7b_mirror \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.use_3drope=True \
    reward_model.use_hybrid_reward_manager=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1
```

#### 3B Model Training

The 3B variant uses larger micro-batch sizes (since the model is smaller) and trains for more epochs:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.system_prompt="tool_crop_mirror" \
    data.train_files=[${BASE_IMAGE_DIR}/rl_train/mmoral-resoning-train-1k.json] \
    data.val_files=[${BASE_IMAGE_DIR}/rl_val_easy/val.json,${BASE_IMAGE_DIR}/rl_val_medium/val.json,${BASE_IMAGE_DIR}/rl_val_hard/val.json] \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.answer_key=solution \
    data.tool_call="crop_mirror" \
    data.general_qa_reward_fn="general_qa_tool_mc" \
    data.gpt_general_qa_reward_fn="general_qa_tool_condition_curiosity" \
    data.gpt_extract_answer=True \
    data.gpt_threads=64 \
    actor_rollout_ref.model.path=/path/to/qwen2_5vl-3b-coldstart-ckpt \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.vllm_infer_batch_size=8 \
    trainer.experiment_name='3b-mirror-train' \
    trainer.default_local_dir=./save/3b_mirror \
    trainer.total_epochs=50 \
    ... # remaining flags identical to the 7B command above
```

Key differences between 3B and 7B:

| Parameter | 7B | 3B |
|---|---|---|
| `ppo_mini_batch_size` | 4 | 8 |
| `ppo_micro_batch_size_per_gpu` | 1 | 2 |
| `rollout.log_prob_micro_batch_size_per_gpu` | 4 | 8 |
| `rollout.vllm_infer_batch_size` | 4 | 8 |
| `gpt_threads` | 80 | 64 |
| `total_epochs` | 1 | 50 |

### 1.6 Validation Commands

Validation runs the trained model in inference-only mode (`trainer.val_only=True`) with greedy decoding (`val_do_sample=False`, `val_n=1`).

#### Validation on Easy / Medium / Hard Splits

```bash
export BASE_IMAGE_DIR="/path/to/your/data"
export API_KEY="your_judge_api_key"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.system_prompt="tool_crop" \
    data.train_files=[${BASE_IMAGE_DIR}/rl_train/train.json] \
    data.val_files=[${BASE_IMAGE_DIR}/rl_val_medium/val.json] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=0 \
    data.tool_call_penalty=0 \
    data.general_qa_reward_fn="general_qa_tool_mc" \
    data.gpt_general_qa_reward_fn="general_qa_tool" \
    data.gpt_extract_answer=True \
    data.extract_answer_tags="strict" \
    data.return_raw_chat=True \
    data.gpt_threads=50 \
    data.tool_call="crop" \
    data.use_tgt_size=False \
    data.max_pixels=2000000 \
    data.min_pixels=40000 \
    reward_model.reward_manager=naive_multithreads_tool \
    actor_rollout_ref.actor.ignore_exceed=True \
    actor_rollout_ref.model.path=/path/to/trained-checkpoint/actor/huggingface \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'"'"'image'"'"': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=12 \
    'actor_rollout_ref.rollout.val_limit_mm_per_prompt={'"'"'image'"'"': 12}' \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
    actor_rollout_ref.rollout.vllm_infer_batch_size=16 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.save_traj=False \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MMOral-Eval' \
    trainer.experiment_name='eval-medium' \
    trainer.val_generations_to_log_to_wandb=512 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=./saves/eval_medium \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.use_3drope=True \
    reward_model.use_hybrid_reward_manager=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    actor_rollout_ref.rollout.val_n=1 \
    actor_rollout_ref.rollout.val_do_sample=False \
    trainer.val_only=True
```

To evaluate on a different split, change `data.val_files` to point to `rl_val_easy/val.json` or `rl_val_hard/val.json`.

#### Validation on OPG Benchmark

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.system_prompt="tool_crop" \
    data.train_files=[${BASE_IMAGE_DIR}/rl_train/train.json] \
    data.val_files=[${BASE_IMAGE_DIR}/rl_val_opg/mmoral-reasoning_opg.json] \
    data.general_qa_reward_fn="general_qa_tool" \
    data.gpt_general_qa_reward_fn="general_qa_tool" \
    data.gpt_threads=20 \
    data.tool_call="crop" \
    actor_rollout_ref.model.path=/path/to/trained-checkpoint \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_generation_round=4 \
    actor_rollout_ref.rollout.limit_mm_per_prompt={image:6} \
    actor_rollout_ref.rollout.val_max_generation_round=4 \
    actor_rollout_ref.rollout.val_limit_mm_per_prompt={image:6} \
    actor_rollout_ref.rollout.vllm_infer_batch_size=8 \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name='eval-opg' \
    trainer.default_local_dir=./saves/eval_opg \
    actor_rollout_ref.rollout.val_n=1 \
    actor_rollout_ref.rollout.val_do_sample=False \
    trainer.val_only=True \
    ... # remaining flags identical to the standard val command
```

Key validation flags:
- `trainer.val_only=True` -- skip training, run validation only
- `actor_rollout_ref.rollout.val_do_sample=False` -- greedy decoding
- `actor_rollout_ref.rollout.val_n=1` -- single rollout per sample
- `trainer.save_freq=-1` -- disable checkpoint saving

### 1.7 LLM Judge Evaluation

After obtaining model predictions, use the standalone LLM judge to score them:

```bash
export API_KEY="your_judge_api_key"

python tools/llm_judge.py \
    --input predictions.json \
    --output scored.json \
    --api-host jeniya.top \
    --judge-model gpt-5-nano-2025-08-07
```

Input format: a JSON list of `{"question", "ground_truth", "prediction"}` records. Output: per-case scores (0.0--1.0) and an overall average.

---

## 2. Data Curation Tools

The `tools/` directory contains the complete **DentalProbe** data-curation pipeline, which converts raw dental datasets into high-quality multi-turn reasoning data for RL training. The pipeline has three stages plus supporting utilities.

### 2.1 Raw Dataset Processors

These scripts convert dataset-specific annotation formats into a unified JSON schema with fields: `image_name`, `image_width`, `image_height`, `source`, `modality`, `Question`, `Answer/Full Answer`, `Precise Grounding Position`, `Contextual Grounding Position`, and `Category`.

| Script | Source Dataset | Annotation Format | Dentition |
|---|---|---|---|
| `data_process_for_Tufts_Dental_Database.py` | Tufts University | Expert/student JSON with polygons | Mixed (Child) |
| `data_process_for_Sichuan_Uni_Children.py` | Sichuan University | LabelMe JSON with base64-encoded images | Mixed (Child) |
| `data_process_for_DENTEX_CHALLENGE_2023.py` | University of Zurich | COCO JSON with 3-level category hierarchy | Permanent (Adult) |
| `data_process_for_Dental_Conditions_Detection_2025_Romania.py` | Iuliu Hatieganu Univ. | COCO JSON with tooth IDs | Permanent (Adult) |

Usage (example with Tufts):

```bash
python tools/data_process_for_Tufts_Dental_Database.py
# Reads expert.json and student.json, outputs processed_annotations.json
```

### 2.2 Format Conversion and Tooth-ID Assignment

**YOLO to COCO conversion:**

```bash
python tools/yolo2coco.py
# Converts YOLO .txt labels to COCO JSON format
# Edit image_folder, label_folder, output_json_file inside the script
```

**DINO-based tooth-ID assignment** -- runs a Swin-L DINO detector to match each disease annotation to the nearest tooth via IoU:

```bash
python tools/infer_dino.py \
    --image /path/to/test/image.jpg \
    --config config/DINO/DINO_5scale_swinL_panoramic_x-ray_32ToothID.py \
    --checkpoint Teeth_Visual_Experts_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth \
    --confidence 0.3 \
    --output ./outputs/
```

### 2.3 DentalProbe Step 0 -- Region Proposal Generation

Generates crop-region proposals from ground-truth bounding boxes via spatial clustering (KMeans on bbox centers), union-find merging (horizontal/vertical overlap + IoM >= 0.7), and controlled expansion.

Each `step_0_data*` script handles a different dataset source:

```bash
# Data1 (Tufts) -- simple bbox clustering + expansion
python tools/step_0_data1_proposal.py \
    --new-json /path/to/Multimodal_data_Tufts.json \
    --out /path/to/proposals_data1.json \
    --expansion 100 \
    --area-threshold 0.15 \
    --visualize \
    --image-root /path/to/images

# Data2 (Sichuan) -- category-aware clustering for pediatric conditions
python tools/step_0_data2_proposal.py \
    --new-json /path/to/Multimodal_data_Sichuan.json \
    --out /path/to/proposals_data2.json \
    --visualize --image-root /path/to/images

# Data3 (DENTEX) -- teeth-ID matching via IoU with per-image teeth JSONs
python tools/step_0_data3_proposal.py \
    --teeth-json-dir /path/to/id_jsons \
    --new-json /path/to/Multimodal_data_DENTEX.json \
    --out /path/to/proposals_data3.json

# Data4 (Romania) -- bone resorption + other lesion handling with teeth matching
python tools/step_0_data4_proposal.py \
    --teeth-json-dir /path/to/id_jsons \
    --new-json /path/to/processed_annotations.json \
    --out /path/to/proposals_data4.json
```

### 2.4 DentalProbe Step 1 -- Multi-Agent VLM Filtering

Uses a multi-agent VLM pipeline (GPT-based) to filter and verify the region proposals. The pipeline performs full-image analysis, cropped-region analysis, comparative analysis (original vs. mirrored contralateral reference), and immediate output verification. Each script handles one dataset:

```bash
export API_KEY="your_api_key"

python tools/step_1_data1_multi_agents_filtering.py \
    --input-json /path/to/proposals_data1.json \
    --source-json /path/to/Multimodal_data_Tufts.json \
    --image-root /path/to/images \
    --output /path/to/filtered_data1.json

# Similarly for data2, data3, data4:
python tools/step_1_data2_multi_agents_filtering.py ...
python tools/step_1_data3_multi_agents_filtering.py ...
python tools/step_1_data4_multi_agents_filtering.py ...
```

The shared utility `step_1_multi_agents_data_processing.py` provides common multi-agent data processing functions.

### 2.5 DentalProbe Step 2 -- GPT-Based Dialogue Rewriting

Rewrites the filtered data into multi-turn reasoning dialogues suitable for RL training. Each script calls a GPT API to generate `<Think>...</Think><Answer>...</Answer>` format dialogues:

```bash
export API_KEY="your_api_key"

# Data1
python tools/step_2_data1_rewrte.py \
    --api-key $API_KEY \
    --input-path /path/to/multi_round_data1.json \
    --output-path /path/to/multi_round_data1_step_2.json \
    --checkpoint-path /path/to/checkpoint.json \
    --model gpt-5-mini-2025-08-07

# Data4 (with disease-definition enrichment)
python tools/step_2_data4_rewrite.py \
    --api-key $API_KEY \
    --input-path /path/to/multi_round_data4.json \
    --output-path /path/to/multi_round_data4_step_2.json

# MMOral data rewriting (with comprehensive disease definitions for all 4 datasets)
python tools/step_2_mmoral_data_rewrite.py \
    --api-key $API_KEY \
    --input-path /path/to/mmoral_data.json \
    --output-path /path/to/mmoral_data_step_2.json
```

Supports sharded processing for parallelism:

```bash
python tools/step_2_data1_rewrte.py \
    --total-shards 4 --current-shard 0 ...
python tools/step_2_data1_rewrte.py \
    --total-shards 4 --current-shard 1 ...
# ... etc.
```

### 2.6 Evaluation and Visualization Utilities

**Keyword-based sample filtering** -- filters large JSON/JSONL files using keyword matching with reservoir sampling:

```bash
python tools/filter_by_keywords.py \
    --input /path/to/r1_mmoral_240k.json \
    --keywords /path/to/opg_keywords_en_dict.txt \
    --output /path/to/output_matched_20k.json \
    --limit 20000 \
    --seed 42
```

**Bounding-box visualization** (Tufts):

```bash
python tools/visual_for_Tufts_Dental_Database.py
# Edit json_path, image_base_dir, output_dir inside the script
```

**Human evaluation bar chart** -- generates the bar chart for human scoring of DentalProbe data quality:

```bash
python tools/draw_bar_chart_for_human_scoring_DentalProbe_data_quality.py
# Outputs: human_scoring_CoT_data_quality_DentalProbe.png
```

---

## 3. Full Pipeline Summary

```
Raw Datasets (4 sources)
    │
    ▼  data_process_for_*.py / yolo2coco.py / infer_dino.py
Standardized Annotation JSONs
    │
    ▼  step_0_data*_proposal.py
Region Proposals (with bbox clustering + expansion)
    │
    ▼  step_1_data*_multi_agents_filtering.py
Filtered Multi-Turn Data (VLM-verified)
    │
    ▼  step_2_data*_rewr*.py
Reasoning Dialogue Data (Think/Answer format)
    │
    ▼  mirror_train_7b.sh / mirror_train_3b.sh
RL-Trained OralGPT-Plus Model
    │
    ▼  val.sh / val_opg.sh / llm_judge.py
Evaluation Results
```
