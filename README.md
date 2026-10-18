<div align="center">
  <img src="assets/mmoral-logo.png" width="150px">
  <img src="assets/logo_OralGPT-Plus.png" width="150px">
  <img src="assets/logo_OralAgent.png" width="150px">
  <img src="assets/logo_OralFlow.png" width="150px">
  <img src="assets/logo_OralDetect.png" width="150px">
  <br><br>
  <p>
    📄&ensp;<a href="https://arxiv.org/abs/2509.09254"><b>[NeurIPS'25]OralGPT</b></a>
    &ensp;·&ensp;
    <a href="https://arxiv.org/abs/2511.22055"><b>[CVPR'26]OralGPT-Omni</b></a>
    &ensp;·&ensp;
    <a href="https://arxiv.org/abs/2603.06366"><b>[CVPR'26]OralGPT-Plus</b></a>
    <br>
    <a href="https://arxiv.org/abs/2605.27378"><b>[arXiv'26]OralAgent</b></a>
    &ensp;·&ensp;
    <a href="https://doi.org/10.1038/s41746-026-03385-9"><b>[npj Digit. Med.'26]PlaqueSAM</b></a>
    &ensp;·&ensp;
    <b>OralFlow</b>
  </p>
  <p>
    🤗&ensp;<a href="https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench"><b>MMOral-OPG-Bench</b></a>
    &ensp;·&ensp;
    <a href="https://huggingface.co/datasets/OralGPT/OralDetect-Bench"><b>OralDetect-Bench</b></a>
    &ensp;·&ensp;
    <a href="https://arxiv.org/abs/2605.27378"><b>OralQA-ZH</b></a>
    <br>
    🤗&ensp;<a href="https://huggingface.co/OralGPT"><b>[Coming]MMOral-Uni-Bench</b></a>
    &ensp;·&ensp;
    <a href="https://huggingface.co/OralGPT"><b>[Coming]OralFlow-Bench</b></a>
  </p>
  <p>
    🚧&ensp;<a href="./OralGPT-X/"><b>OralGPT-X</b></a>
    &ensp;·&ensp;
    <a href="./OralDetect/"><b>OralDetect</b></a>
    &ensp;·&ensp;
    <b>OralClue</b>
    &ensp;·&ensp;
    <b>OralRSI</b>
    &ensp;·&ensp;
    <b>OralGPT-3D</b>
    &ensp;·&ensp;
    <a href="./OralGPT-V/"><b>OralGPT-V</b></a>
  </p>
</div>

# OralGPT Family 🦷

**OralGPT** is a family of dental AI **models** and **benchmarks** for digital dentistry, covering multimodal large language models, tool-using agents, and clinical evaluation.

- **Researchers**: papers, benchmarks, and an evaluation toolkit.
- **Developers**: in-repo training and evaluation entry points, plus released weights on [Hugging Face](https://huggingface.co/OralGPT).
- **Collaborators**: contact below.

## Table of Contents

- [News](#news)
- [Models](#models)
- [Benchmarks](#benchmarks)
- [Repository](#repository)
- [Citation](#citation)
- [Contact](#contact)

---

## News

- **[2026-10-11]** 🎉 **PlaqueSAM** has been published in **npj Digital Medicine**: [Photography-based dental plaque detection and report generation among preschool children using PlaqueSAM](https://www.nature.com/articles/s41746-026-03385-9). Code: [isjinghao/PlaqueSAM](https://github.com/isjinghao/PlaqueSAM).
- **[2026-09-01]** 🔥 **OralFlow** is coming soon...
- **[2026-08-29]** 🔥🔥 **OralDetect**, an open-vocabulary dental detector spanning 5 imaging modalities, has been released: weights on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralDetect-Family), with the [training set](https://huggingface.co/datasets/OralGPT/OralDetect-Training) and [benchmark](https://huggingface.co/datasets/OralGPT/OralDetect-Bench). 👏 Welcome to detect on your own dental images.
- **[2026-04-09]** 🔥 **OralAgent** has been released on [arXiv](https://arxiv.org/abs/2605.27378).
- **[2026-03-27]** 🔥 **[NeurIPS 2025] MMOral‑OPG‑Bench** is now supported on [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/mmoral_opg_open.py) for evaluation.
- **[2026-03-02]** 🚀 **OralGPT-Captioning-4B-Base** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Captioning-4B-Base). 👏 Welcome to generate visual description of dental images.
- **[2026-02-22]** 🎉 Our papers **[OralGPT-Omni](https://arxiv.org/abs/2511.22055)** and **OralGPT-Plus** have been accepted by **CVPR 2026**.
- **[2025-12-17]** 🔥🔥🔥 **OralGPT-Omni-7B-Instruct** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct). 👏 Welcome to try it.
- **[2025-11-27]** 🔥🔥🔥 Our paper of **OralGPT-Omni** has been released on [arXiv](https://arxiv.org/abs/2511.22055).
- **[2025-10-31]** 🔥 **[NeurIPS 2025] MMOral-Bench** (MMOral-OPG-Bench) has been released on 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench). 👏 Welcome to evaluate your MLLMs.
- **[2025-09-22]** 🚀 Multiple **Dental Visual Expert Models** have been released on 🤗 [Hugging Face](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models).
- **[2025-09-19]** 🎉 Our paper of **OralGPT** has been accepted by **NeurIPS 2025**.
- **[2025-09-11]** 🎉 Our paper of **OralGPT** has been released on [arXiv](https://arxiv.org/abs/2509.09254).
- 🤝 For collaboration inquiries, please contact us at: 📮 isjinghao@gmail.com

---

## Models

Specialist and general models in the OralGPT family. PlaqueSAM lives in its own repository.

| Model | Description | Paper | Code / weights |
| --- | --- | --- | --- |
| **OralGPT** | Dental MLLM for panoramic X-ray analysis, with chain-of-thought diagnostic reasoning | [NeurIPS 2025](https://arxiv.org/abs/2509.09254) | [MMOral-OPG](./MMOral-OPG/) |
| **OralGPT-Omni** | Versatile dental MLLM across intraoral photos and video, radiographs, histopathology, and text | [CVPR 2026](https://arxiv.org/abs/2511.22055) | [HF](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct) · [MMOral-Omni](./MMOral-Omni/) |
| **OralGPT-Plus** | Learns to use visual tools via reinforcement learning for panoramic X-ray analysis | [CVPR 2026](https://arxiv.org/abs/2603.06366) | [OralGPT-Plus](./OralGPT-Plus/) |
| **OralAgent** | Interactive dental image analysis with reasoning, tools, and knowledge | [arXiv 2026](https://arxiv.org/abs/2605.27378) | [Paper](https://arxiv.org/abs/2605.27378) |
| **OralGPT-CMF** | Fine-tuned multimodal model for oral and maxillofacial cases | Ongoing | [HF](https://huggingface.co/OralGPT/OralGPT-CMF-v1) · [OralGPT-CMF](./OralGPT-CMF/) |
| **OralDetect** | Open-vocabulary detector across five imaging modalities | Released | [HF](https://huggingface.co/OralGPT/OralDetect-Family) · [OralDetect](./OralDetect/) |
| **PlaqueSAM** | Plaque detection and visual report generation from smartphone photos of preschool children | [npj Digital Medicine 2026](https://doi.org/10.1038/s41746-026-03385-9) | [GitHub](https://github.com/isjinghao/PlaqueSAM) |
| **OralGPT-X** | Oral image generation and translation for CBCT, MRI, and orthognathic imaging | Ongoing | [OralGPT-X](./OralGPT-X/) |
| **OralGPT-V** | Oral world model | In progress | [OralGPT-V](./OralGPT-V/) |

Supported imaging and text inputs across the family include intraoral images and videos, photographs, panoramic X-rays, periapical radiographs, cephalometric radiographs, histopathological images, and textual question answering.

---

## Benchmarks

Clinical evaluation sets. Run MMOral language-model benchmarks with [MMOral-Bench-EvalKit](./MMOral-Bench-EvalKit/).

| Benchmark | What it evaluates | Status | Link |
| --- | --- | --- | --- |
| **MMOral-OPG-Bench** | Open- and closed-ended panoramic X-ray understanding. Items are reviewed by clinical dentists. | Released | [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench) · [EvalKit](./MMOral-Bench-EvalKit/) |
| **MMOral-Omni-Bench** | Multi-modality dental understanding | Coming | [EvalKit](./MMOral-Bench-EvalKit/) |
| **MMOral-Uni-Bench** | Unified dental MLLM evaluation | Coming | [Hugging Face org](https://huggingface.co/OralGPT) |
| **OralDetect-Bench** | Open-vocabulary detection on five modalities | Released | [Hugging Face](https://huggingface.co/datasets/OralGPT/OralDetect-Bench) |
| **OralQA-ZH** | Chinese multiple-choice questions across 11 oral subspecialties, introduced with OralAgent | Released | [OralAgent](https://arxiv.org/abs/2605.27378) |
| **OralFlow-Bench** | Longitudinal multimodal oral and maxillofacial trajectories | Coming | [OralFlow](./OralFlow/) |
| **OralGPT-X-Bench** | CBCT restoration, orthognathic simulation, and MRI translation | In repo | [OralGPT-X](./OralGPT-X/oralgpt_x_bench/) |

Evaluation steps for MMOral-OPG-Bench are in [MMOral-Bench-EvalKit](./MMOral-Bench-EvalKit/).

---

## Repository

| Path | Role |
| --- | --- |
| [MMOral-Bench-EvalKit](./MMOral-Bench-EvalKit/) | Evaluation toolkit for MMOral-OPG-Bench, MMOral-Omni-Bench, and newer splits |
| [MMOral-OPG](./MMOral-OPG/) | Panoramic X-ray data generation (attributes, chat, reports, VQA) |
| [MMOral-Omni](./MMOral-Omni/) | Multi-modality data for OralGPT-Omni |
| [OralGPT-Plus](./OralGPT-Plus/) | OralGPT-Plus training and DentalProbe data tools |
| [OralGPT-CMF](./OralGPT-CMF/) | OralGPT-CMF inference |
| [OralGPT-X](./OralGPT-X/) | Generation data scripts and OralGPT-X-Bench |
| [OralGPT-V](./OralGPT-V/) | Oral world model |
| [OralDetect](./OralDetect/) | Open-vocabulary detection training and evaluation |
| [OralFlow](./OralFlow/) | OralFlow-Bench construction pipeline |
| [LLaMA-Factory](./LLaMA-Factory/) | Upstream training framework, vendored for convenience |
| [PlaqueSAM](https://github.com/isjinghao/PlaqueSAM) | Separate repository |

---

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{hao2026plaquesam,
  title={Photography-based dental plaque detection and report generation among preschool children using PlaqueSAM},
  author={Hao, Jing and Guo, Kaixin and Shan, Zhiyi and Yang, Yanqi and Tsoi, James Kit Hon and Lam, Phoebe Pui Ying and Hung, Kuo Feng},
  journal={npj Digital Medicine},
  year={2026},
  doi={10.1038/s41746-026-03385-9}
}
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

## Contact

- 📮 **Email**: isjinghao@gmail.com
