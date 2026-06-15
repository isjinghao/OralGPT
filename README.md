<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/mmoral-logo.png" width="150px">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/logo_OralGPT-Plus.png" width="150px">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/logo_OralAgent.png" width="150px">
  <br><br>

  <!-- 链接部分 -->
  <a href="https://arxiv.org/pdf/2509.09254" target="_blank">📄 <b>[NeurIPS'25] OralGPT</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2511.22055" target="_blank">📄 <b>[CVPR'26] OralGPT-Omni</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2603.06366" target="_blank">📄 <b>[CVPR'26] OralGPT-Plus</b></a> &nbsp;|&nbsp;
  
  <a href="https://arxiv.org/abs/2605.27378" target="_blank">📄 <b>[Arxiv'26] OralAgent</b></a> 

    
  <a href="https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench" target="_blank">
    🤗 <b>MMOral-OPG-Bench</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/OralGPT" target="_blank">
    🤗 <b>MMOral-Uni-Bench (Coming soon)</b>
  </a>
</div>

# OralGPT Family 🦷🦷

**OralGPT** is a **family of dental multimodal large language models (MLLMs)** and benchmarks for **AI-driven digital dentistry**.

- **For researchers**: benchmark + evaluation toolkit + papers.
- **For developers**: ready-to-run evaluation pipeline + configs.
- **For collaborators**: contact below.

## 📌 Quick Links

- **Papers**: [OralGPT](https://arxiv.org/abs/2509.09254) · [OralGPT-Omni](https://arxiv.org/abs/2511.22055) · [OralGPT-Plus](https://arxiv.org/abs/2603.06366) · [OralAgent](https://arxiv.org/abs/2605.27378) 
- **Datasets/Models**: [MMOral-OPG-Bench](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench) · [Hugging Face Org](https://huggingface.co/OralGPT)
- **Evaluation**: [MMOral-Bench-EvalKit](./MMOral-Bench-EvalKit/) for MMOral-OPG-Bench and MMOral-Omni-Bench

## 📖 Table of Contents

- [News](#news)
- [Overview](#overview)
- [Getting Started (MMOral-OPG-Bench)](#getting-started-mmoral-opg-bench)
- [Repository Structure](#repository-structure)
- [MMOral-Bench](#mmoral-bench)
- [Citation](#citation)
- [Contact](#contact)

---

## 🔔 News
- **[2026-04-09]** 🔥 **OralAgent** has been released on [arXiv](https://arxiv.org/abs/2605.27378).  
- **[2026-03-27]** 🔥 **[NeurIPS 2025] MMOral‑OPG‑Bench** is now supported on [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/mmoral_opg_open.py) for evaluation.
- **[2026-03-02]** 🚀  **OralGPT-Captioning-4B-Base** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Captioning-4B-Base). 👏 Welcome to generate visual description of dental images. 
- **[2026-02-22]** 🎉 Our paper of **[OralGPT-Omni](https://arxiv.org/abs/2511.22055)** and **OralGPT-Plus** have been accepted by **CVPR 2026**.
- **[2025-12-17]** 🔥🔥🔥 **OralGPT-Omni-7B-Instruct** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct). 👏 Welcome to try it.
- **[2025-11-27]** 🔥🔥🔥 Our paper of **OralGPT-Omni** has been released on [arXiv](https://arxiv.org/abs/2511.22055).  
- **[2025-10-31]** 🔥 **[NeurIPS 2025] MMOral-Bench** (MMOral-OPG-Bench) has been released on 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench). 👏 Welcome to evaluate your MLLMs.
- **[2025-09-22]** 🚀 Multiple **Dental Visual Expert Models** have been released on 🤗 [Hugging Face](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models). 
- **[2025-09-19]** 🎉 Our paper of **OralGPT** has been accepted by **NeurIPS 2025**. 
- **[2025-09-11]** 🎉 Our paper of **OralGPT** has been released on [arXiv](https://arxiv.org/abs/2509.09254).  
- 🔜 We are actively developing **MMOral-Bench v2**, which will include:  
  - ✅ More dental imaging modalities  
  - ✅ Professional dentistry exam questions  
  - ✅ Comprehensive evaluation of multiple MLLM performance in digital dentistry
- 🤝 For collaboration inquiries, please contact us at: 📮 isjinghao@gmail.com

---

## ✨ Overview

**OralGPT** is a **series multimodal large language model (MLLM) specialized in digital dentistry**. It supports diverse dental imaging modalities, including:  

- Intraoral images & videos  
- Photographs  
- Panoramic X-rays 
- Periapical radiographs  
- Cephalometric radiographs  
- Histopathological images  
- Textual Question & Conversation  

OralGPT aims to be the foundation MLLM for AI-driven digital dentistry — bridging multimodal reasoning with clinical expertise. With **Chain-of-Thought (CoT) reasoning**, OralGPT simulates the diagnostic process of radiologists, ensuring outputs that are **interpretable, trustworthy, and clinically reliable**.  

---

## 🚀 Getting Started (MMOral-OPG-Bench)

This is the fastest way to run an evaluation with the provided toolkit.

### 1) Prepare API credentials

Create a `.env` file under `MMOral-Bench-EvalKit/` and fill in your credentials:

```bash
OPENAI_API_KEY=
OPENAI_API_BASE=
```

> **Tip:** `.env` contains private keys. Do **not** commit it to any public repo.

### 2) Configure models & judge

Edit the evaluation config (example file name used by the toolkit: `config_mmoral_opg.json`) to specify:

- models to evaluate
- benchmark split (e.g., `MMOral_OPG_CLOSED`, `MMOral_OPG_OPEN`)
- judge model (e.g., `gpt-4-turbo` or `gpt-5-mini`)

See: `MMOral-Bench-EvalKit/README.md` for full examples.

### 3) Run evaluation

From `MMOral-Bench-EvalKit/`, run:

```bash
python run.py --config config_mmoral_opg.json \
  --mode all \
  --api-nproc 8 \
  --work-dir '.' \
  --verbose
  # --reuse
```

Add `--reuse` to resume existing results.

---

## 🧭 Repository Structure

- **`MMOral-Bench-EvalKit/`**: unified evaluation toolkit for MMOral-OPG-Bench, MMOral-Omni-Bench, and newer benchmark variants.
- **`MMOral-Omni/`**: OralGPT-Omni related materials (in progress).
- **`LLaMA-Factory/`**: training framework (upstream project included for convenience).

## 📏 MMOral-Bench  

Currently, you can evaluate your MLLM’s performance on **panoramic X-ray analysis** using **MMOral-Bench**.  
All benchmark data are **reviewed and validated by professional clinical dentists**, ensuring **accuracy and clinical reliability**.  

### Performance
<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/MMOral-OPG-Bench-Performance.jpg">
</div>

### Evaluation of MMOral-Bench

Our benchmark consists of both Open-Ended and Closed-Ended evaluation formats, with corresponding TSV files available at 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench).

For benchmark evaluation, please use the unified [**MMOral-Bench-EvalKit**](./MMOral-Bench-EvalKit/) toolkit. It supports both MMOral-OPG-Bench splits:

- `MMOral_OPG_CLOSED` for closed-ended multiple-choice evaluation.
- `MMOral_OPG_OPEN` for open-ended evaluation with an LLM judge.

The same evaluation folder also supports MMOral-Omni-Bench and newer benchmark variants. To evaluate a model, edit the `data` field in `MMOral-Bench-EvalKit/config_mmoral_opg.json` or create a config with either `MMOral_OPG_CLOSED` or `MMOral_OPG_OPEN`, then run `python run.py --config <config>.json --mode all`.

## 📌 Citation  

If you find our work helpful, please cite us:  

```bibtex
@article{hao2026oralagent,
  title={OralAgent: Integrating Reasoning, Tools, and Knowledge for Interactive Dental Image Analysis},
  author={Hao, Jing and Dai, Siyuan and Zhang, Yongxin and Liang, Yuci and Wu, Jiamin and Bao, Jiahao and Fan, Yuxuan and Ye, Zanting and Sun, Yanpeng and Zhang, Xinyu and others},
  journal={arXiv preprint arXiv:2605.27378},
  year={2026}
}
@article{hao2025oralgpt-omni,
  title={OralGPT-Omni: A Versatile Dental Multimodal Large Language Model},
  author={Hao, Jing and Liang, Yuci and Lin, Lizhuo and Fan, Yuxuan and Zhou, Wenkai and Guo, Kaixin and Ye, Zanting and Sun, Yanpeng and Zhang, Xinyu and Yang, Yanqi and others},
  journal={CVPR 2026},
  year={2025}
}
@article{fan2026oralgpt-plus,
  title={OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis},
  author={Fan, Yuxuan and Hao, Jing and Chen, Hong and Bao, Jiahao and Shao, Yihua and Liang, Yuci and Hung, Kuo Feng and Tang, Hao},
  journal={CVPR 2026},
  year={2026}
}
@article{hao2025mmoral,
  title={Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author={Hao, Jing and Fan, Yuxuan and Sun, Yanpeng and Guo, Kaixin and Lin, Lizhuo and Yang, Jinrong and Ai, Qi Yong H and Wong, Lun M and Tang, Hao and Hung, Kuo Feng},
  journal={NeurIPS 2025},
  year={2025}
}
@article{hao2025oraldataset,
  title={Characteristics, licensing, and ethical considerations of openly accessible oral-maxillofacial imaging datasets: a systematic review},
  author={Hao, Jing and Nalley, Andrew and Yeung, Andy Wai Kan and Tanaka, Ray and Ai, Qi Yong H and Lam, Walter Yu Hang and Shan, Zhiyi and Leung, Yiu Yan and AlHadidi, Abeer and Bornstein, Michael M and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={412},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## 🤝 Contact

- 📮 **Email**: isjinghao@gmail.com
