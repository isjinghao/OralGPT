# OralGPT-V

**OralGPT-V** 是 [OralGPT Family](https://github.com/isjinghao/OralGPT) 中面向 **口腔世界模型（Oral World Model）** 的研究方向。本项目刚刚启动，后续代码与实验将集中在本目录维护。

---

## 什么是口腔世界模型？

**世界模型（World Model）** 指能够内化环境动态规律的模型：给定当前状态与动作，预测未来状态如何演化。在通用 AI 领域，世界模型已被用于机器人控制、视频生成、具身智能等任务，使智能体可以在"想象"中进行规划与推理，而不仅依赖对当前观测的被动响应。

**口腔世界模型（Oral World Model）** 将这一思想引入 **数字口腔医学** 场景，目标是构建能够理解并预测口腔环境演化的多模态模型，例如：

- **影像演化预测**：给定当前口内照片、全景片或 CBCT，预测治疗干预（如正畸、修复、种植）后的形态变化
- **疾病进展建模**：基于历史影像与临床信息，推演龋齿、牙周病、骨吸收等病变的时序发展
- **治疗过程仿真**：在虚拟环境中模拟拔牙、植骨、正畸施力等操作对软硬组织的影响
- **跨模态状态推断**：从 2D 影像推断 3D 解剖结构，或从静态快照重建动态咀嚼、开口等功能状态

与 OralGPT 系列中侧重 **理解与问答** 的多模态大模型（MLLM）不同，OralGPT-V 更关注 **预测与生成**——让模型不仅"看懂"口腔影像，还能"想象"口腔世界如何变化。

---

## 研究动机

数字口腔医学的核心挑战之一，是临床决策往往依赖对 **未来状态** 的判断：正畸方案是否可行？种植体位置是否合适？牙周治疗能否阻止骨丧失？传统方法依赖医生的经验与静态影像，缺乏对动态演化的系统性建模。

OralGPT-V 希望借助世界模型，为数字口腔提供：

| 能力 | 说明 |
|------|------|
| **前瞻性推理** | 从当前观测推演未来可能的状态，辅助治疗规划 |
| **反事实分析** | 对比"若采取不同治疗方案，结果会如何" |
| **数据高效学习** | 从有限配对/时序数据中学习口腔环境的隐含动态 |
| **与 MLLM 协同** | 与 OralGPT-Omni、OralGPT-Plus 等理解型模型形成"理解 + 预测"的完整链路 |

---

## 与 OralGPT Family 的关系

```
OralGPT Family
├── OralGPT / OralGPT-Omni    → 多模态理解与推理（MLLM）
├── OralGPT-Plus              → 视觉工具调用与强化学习
├── OralGPT-CMF               → 颅颌面多模态分析
├── OralGPT-X                 → 正颌外科等专项任务
└── OralGPT-V  ← 本项目       → 口腔世界模型：预测、生成与仿真
```

OralGPT-V 并非替代现有 OralGPT 模型，而是在其基础上拓展 **生成式与预测式** 能力，使 OralGPT 生态从"看懂口腔"走向"预见口腔"。

---

## 项目状态

> 🚧 **项目刚启动，代码与文档持续更新中。**

当前阶段主要进行方向梳理与基础架构搭建，后续将逐步开放：

- [ ] 数据管线与预处理脚本
- [ ] 基线世界模型实现
- [ ] 训练与评估流程
- [ ] 预训练权重与 Demo

---

## 目录结构（规划中）

```
OralGPT-V/
├── README.md          # 本文件
├── data/              # 数据准备与预处理（待添加）
├── models/            # 模型定义（待添加）
├── training/          # 训练脚本（待添加）
├── evaluation/        # 评估与可视化（待添加）
└── configs/           # 实验配置（待添加）
```

---

## 引用

若本工作对您有帮助，请同时引用 OralGPT 系列相关论文：

```bibtex
@article{hao2025oralgpt-omni,
  title={OralGPT-Omni: A Versatile Dental Multimodal Large Language Model},
  author={Hao, Jing and Liang, Yuci and Lin, Lizhuo and Fan, Yuxuan and Zhou, Wenkai and Guo, Kaixin and Ye, Zanting and Sun, Yanpeng and Zhang, Xinyu and Yang, Yanqi and others},
  journal={CVPR 2026},
  year={2025}
}
```

---

## 联系

- 📮 **Email**: isjinghao@gmail.com
- 🔗 **OralGPT**: [github.com/isjinghao/OralGPT](https://github.com/isjinghao/OralGPT)
