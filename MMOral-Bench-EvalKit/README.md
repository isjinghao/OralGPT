# 🧠 MMOral Benchmark Evaluation

## ☀️ 0. Benchmarks supported:

1. 🦷 MMOral-OPG-Bench <a href="https://arxiv.org/pdf/2509.09254" target="_blank"><b>[Paper]</b></a>
2. ⚕️ MMOral-Omni-Bench <a href="" target="_blank"><b>[Coming soon]</b></a>

This is the unified evaluation toolkit for the current MMOral benchmarks. It supports both MMOral-OPG-Bench splits:

- `MMOral_OPG_CLOSED`: closed-ended multiple-choice evaluation.
- `MMOral_OPG_OPEN`: open-ended evaluation with an LLM judge.

> You will need access to the gpt-4-turbo or gpt-5-mini as the judge model in the evaluation process.

---


## 🚀 1. Environment Setup

### 🧩 Create a `.env` file

Inside the `MMOral-Bench-EvalKit/` directory, create a `.env` file and fill in your OpenAI API credentials:

```bash
OPENAI_API_KEY=
OPENAI_API_BASE=
```

> 💡 **Important:** The `.env` file stores private API configurations. **Do not upload it** to any public repository!

---

## ⚙️ 2. Judge Model Configuration

Edit the model configuration in `vlmeval/config.py`.
Example configuration:

```python
# VLMEvalKit uses requests.post, so ensure the API base supports POST requests

from functools import partial
from vlmeval.vlm import GPT4V

test_models = {
    "gpt-5-mini": partial(
        GPT4V,
        model="gpt-5-mini",
        api_base="https://www.dmxapi.cn/v1/chat/completions",
        temperature=1,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=True,
    ),
}
```
---

## 🔍 3. Verify Judge Model Configuration

To confirm your judge model is properly configured, run:

```bash
vlmutil check gpt-5-mini
```

> ✅ If the model passes the check and returns a valid response, setup is successful.

---

## ⚒️ 4. Prepare the Evaluation Configuration

Create or edit the file `config_mmoral_opg.json`. This file defines:

- The VLM models to be evaluated
- The benchmark name (e.g., '**MMOral_OPG_CLOSED**', '**MMOral_OPG_OPEN**', '**MMOral_OMNI**', etc.)
- The Judge model (e.g., 'gpt-5-mini' or 'gpt-4-turbo')

Use `MMOral_OPG_CLOSED` for closed-ended evaluation:

```json
{
    "model": {
        "GLM-4.1V-9B-Thinking": {
            "class": "GPT4V",
            "model": "GLM-4.1V-9B-Thinking",
            "temperature": 0.8,
            "img_detail": "high",
            "api_base": "https://www.dmxapi.cn/v1/chat/completions",
            "retry": 10,
            "verbose": true
        }
    },
    "data": {
        "MMOral_OPG_CLOSED": {
            "class": "MMOral_OPG_CLOSED",
            "dataset": "MMOral_OPG_CLOSED"
        }
    },
    "judger": {
        "gpt-4-turbo": {
            "class": "GPT4V",
            "model": "gpt-4-turbo",
            "api_base": "https://www.dmxapi.cn/v1/chat/completions",
            "temperature": 1.0,
            "retry": 10,
            "verbose": true
        }
    }
}

```

Use `MMOral_OPG_OPEN` for open-ended evaluation:

```json
{
    "model": {
        "OralGPT-Omni": {
            "class": "GPT4V",
            "model": "OralAgent",
            "temperature": 0.0,
            "img_detail": "high",
            "api_base": "http://0.0.0.0:8124/v1/chat/completions",
            "retry": 10,
            "timeout": 600,
            "max_tokens": 8192,
            "verbose": true
        }
    },
    "data": {
        "MMOral_OPG_OPEN": {
            "class": "MMOral_OPG_OPEN",
            "dataset": "MMOral_OPG_OPEN"
        }
    },
    "judger": {
        "gpt-5-mini": {
            "class": "GPT4V",
            "model": "gpt-5-mini",
            "img_detail": "high",
            "api_base": "https://www.dmxapi.cn/v1/chat/completions",
            "temperature": 0.0,
            "retry": 10,
            "verbose": true
        }
    }
}
```

Use `MMOral_OMNI` or `MMOral_OMNI_Mini` for Omni-Bench evaluation:

```json
{
    "model": {
        "OralGPT-Omni": {
            "class": "GPT4V",
            "model": "OralAgent",
            "temperature": 0.0,
            "img_detail": "high",
            "api_base": "http://0.0.0.0:8124/v1/chat/completions",
            "retry": 10,
            "timeout": 600,
            "max_tokens": 32768,
            "verbose": true
        }
    },
    "data": {
        "MMOral_OMNI": {
            "class": "MMOral_OMNI",
            "dataset": "MMOral_OMNI",
            // "categories": ["IV", "CE", "PA", "PI"]
        }
    },
    "judger": {
        "gpt-5-mini": {
            "class": "GPT4V",
            "model": "gpt-5-mini",
            "img_detail": "high",
            "api_base": "https://www.dmxapi.cn/v1/chat/completions",
            "temperature": 0.0,
            "retry": 10,
            "verbose": true
        }
    }
}
```


---

## 🧭 5. Start Evaluation
Run the following script to start the evaluation:

```bash
python run.py --config config_mmoral_opg.json \
  --mode all \
  --api-nproc 8 \
  --work-dir '.' \
  --verbose
  # --reuse
```
> 💡 Add `--reuse` if you want to resume the existing evaluation results.

#### 🧹 Optional: Post-processing Model Outputs

If you wish to clean model responses (e.g., remove “thinking” reasoning parts and keep only final answers), edit the post-processing logic in: `MMOral-Bench-EvalKit/vlmeval/inference.py`.

---

## 💬 7.  Feedback & Contributions

Contributions and feedback are highly welcome!

- 🐛 Report Issues via GitHub Issues
- 💡 Submit Pull Requests for improvements
- ⭐ Pls feel free to 📮 isjinghao@gmail.com



## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

If you use VLMEvalKit in your research or wish to refer to published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
@article{oralgpt2025,
  title={Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author={Hao, Jing and Fan, Yuxuan and Sun, Yanpeng and Guo, Kaixin and Lin, Lizhuo and Yang, Jinrong and Ai, Qi Yong H and Wong, Lun M and Tang, Hao and Hung, Kuo Feng},
  journal={arXiv preprint arXiv:2509.09254},
  year={2025}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>
