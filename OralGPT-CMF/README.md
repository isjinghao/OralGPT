# OralGPT-CMF

This repository contains scripts and instructions for testing the fine-tuned
OralGPT-CMF multimodal model:

https://huggingface.co/OralGPT/OralGPT-CMF-v1

## 1. Environment

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda create -n llmf python=3.11 -y
conda activate llmf
pip install -e ".[torch]"
pip install -U gradio huggingface_hub
```

If the Hugging Face model is private, login first:

```bash
hf auth login
hf auth whoami
```

## 2. Download The Model

You can let Hugging Face download the model automatically by using the repo id:

```bash
export MODEL_PATH=OralGPT/OralGPT-CMF-v1
```

Or download it explicitly:

```bash
hf download OralGPT/OralGPT-CMF-v1 \
  --local-dir $HOME/models/OralGPT-CMF-v1

export MODEL_PATH=$HOME/models/OralGPT-CMF-v1
```

Use the same template as training:

```bash
export TEMPLATE=qwen3_vl_nothink
export IMAGE_MAX_PIXELS=131072
```

## 3. Single-Image WebChat With LLaMA-Factory

LLaMA-Factory's built-in `webchat` is convenient for visual testing, but this UI
usually supports one uploaded image at a time.

```bash
cd /root/workspace/LLaMA-Factory
conda activate llmf

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=6006

llamafactory-cli webchat \
  --model_name_or_path OralGPT/OralGPT-CMF-v1 \
  --template qwen3_vl_nothink \
  --trust_remote_code true \
  --infer_backend huggingface \
  --image_max_pixels 131072
```

Open `http://localhost:6006` if running locally. If running on a remote server,
forward or expose port `6006`, then upload one image and ask a question in the
chat box.

If you downloaded the model locally, replace the model path:

```bash
--model_name_or_path $HOME/models/OralGPT-CMF-v1
```
