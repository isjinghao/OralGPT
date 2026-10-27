# OralGPT-CMF

原始数据处理、Qwen3.5-9B微调、测试集推理与AI评分。以下命令使用当前服务器目录和已安装环境。

## 目录职责

| 目录或文件 | 用途 |
| --- | --- |
| `pipeline/` | 原始影像、记录处理及问答数据构造 |
| `sft/data_preparation/` | 数据划分、患者上下文及训练样本准备 |
| `sft/configs/`、`sft/train.sh`、`sft/tools/` | 训练配置、启动、长度检查及曲线工具 |
| `sft/data/`、`sft/models/` | 训练数据、适配器、checkpoint及训练日志 |
| `answer/` | API和本地模型答案生成、续跑及进度查询 |
| `scoring/` | 评分入口、Judge、指标汇总及专用`prompts/` |
| `common/`、`prompts/` | 共用的对话历史、题目路由及提示词 |
| `.env`、`VQA140_final.json`、`VQA387_final.json` | 共享凭据配置与原始VQA数据 |
| `output/pipeline/` | 构造流水线的中间和最终产物 |
| `output/` | 答案、缓存、评分结果与生成/评分日志 |

训练与生成共享对话格式，但训练历史使用参考答案、推理历史使用模型自己的预测，二者不混用。

## 原始数据处理（按需）

原图位于相邻的`OralFlow/SH9HCMFdata`。先设置`OPENAI_API_KEY`；全景片步骤还需要启动本地影像代理。阶段0会处理图片，阶段1–4快捷入口默认覆盖旧输出，已有结果不要直接重跑。

```bash
cd /root/autodl-tmp/OralGPT/OralGPT-CMF
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cmfbench
bash pipeline/run_stage0.sh
bash pipeline/run_stage1.sh
bash pipeline/run_stage2.sh
bash pipeline/run_xray_report.sh
bash pipeline/run_stage3.sh
bash pipeline/run_stage4.sh
python pipeline/build_dataset.py --input-dir output/pipeline --output-file output/pipeline/cmf_sft_dataset.json
```

## SFT数据与训练

先准备根目录的`VQA140_final.json`、`VQA387_final.json`及原图。以下命令均在项目根目录执行：

```bash
cd /root/autodl-tmp/OralGPT/OralGPT-CMF
export PATH=/root/autodl-tmp/venvs/qwen35_sft/bin:$PATH
```

仅在需要重建数据时运行下面两条：第一条重建`sft/data/turn_level`；第二条要求目标目录不存在。已有数据可直接跳到训练。

```bash
python sft/data_preparation/turn_level.py
python sft/data_preparation/question_rules.py --source sft/data/turn_level --output sft/data/question_rules --question-prompts prompts/oral_question_prompts.yaml
```

训练v3（新实验先修改配置的`output_dir`；继续训练使用新位置的`resume_from_checkpoint`）：

```bash
bash sft/train.sh sft/configs/qwen35_lora_v3.yaml
```

配置`qwen35_lora_v1.yaml`、`qwen35_lora_v2.yaml`、`qwen35_lora_v3.yaml`保留各自的训练参数。模型与checkpoint位于`sft/models/`，模型子目录名不变。

## 答案生成

本地v3模型，单分片后台运行；重复执行相同命令可续跑。最后的`1`是分片数，增加时每个分片会加载一份模型。

```bash
bash answer/start_local.sh qwen3.5-9b-v3 "$PWD/sft/models/qwen3_5_9b_lora_turn_level_v3" 4096 1 --dataset "$PWD/sft/data/question_rules/oralcmf_sft_test_rollout.json" --question-prompts "$PWD/sft/data/question_rules/question_prompts.yaml"
bash answer/progress.sh
```

API模型测试：在根目录`.env`配置`ANSWER_OPENAI_API_KEY`和`ANSWER_OPENAI_BASE_URL`，然后运行：

```bash
bash answer/run_api.sh --answer-model gpt-5.5 --num-workers 4
```

本地和API的Python入口分别为`answer/generate_local.py`和`answer/generate_api.py`。缓存位置及模型名称不变，只将原产物目录整体上移到根目录`output/`。

## AI评分

同样读取根目录`.env`，默认标准为`scoring/prompts/oral_scoring.yaml`。修改标准或答案后使用新的`--run-name`。新规则将无参考支持的相关断言纳入评价，尚未在临床维度扣分的漏答、无依据新增、重复和离题合计附加扣0–20分；逐题保留临床原分、附加扣分、双侧引文及最终分。仅确认HTTP中转可信时保留`--allow-http`；HTTPS去掉此项。

```bash
python scoring/score_answers.py --answer-model qwen3.5-9b-v3 --judge-model gpt-5-mini --run-name my_score_run --num-workers 3 --allow-http
```

不请求API的路径与题目数量检查：

```bash
python scoring/score_answers.py --answer-model qwen3.5-9b-v3 --dry-run
```

答案位于`output/<group>/<patient>/evaluation/<model>/answers.json`；逐题评分和患者汇总位于同级`scoring/<judge>/<run>/`，整批汇总位于`output/scoring/<model>/<judge>/<run>/summary.json`。日志位于`output/logs`。这些产物和API密钥均不提交到Git。
