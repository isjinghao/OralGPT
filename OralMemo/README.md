# OralMemBench

面向口腔颌面外科（OralCMF）的多模态Benchmark。以患者多轮多模态问诊数据为输入，产出：

1. **分阶段临床轨迹**（按真实就诊顺序释放各模态证据）；
2. **原子证据**（用 LLM 从可见问答中抽取的最小事实单元）；
3. **证据图**（阶段内 / 跨阶段的依赖关系，含可视化 HTML）；
4. **Benchmark 任务**（感知、回忆、跨模态推理、记忆更新、诊断、治疗）及**评分 rubric**。

---

## 一、目录结构

```
OralMemo/
├── __init__.py                       # 包说明
├── config.py                         # 配置：路径 + .env / OpenAI 设置
├── llm_client.py                     # OpenAI 兼容客户端（限流重试 + JSON 解析）
├── batch_utils.py                    # 多病人选择、输出路径与并行执行
├── scripts/                          # WSL 批量运行入口（自动激活 cmfbench）
│   ├── run_step1_step2.sh
│   ├── run_step3.sh
│   ├── run_step4.sh
│   ├── run_perception_trajectory.sh
│   ├── run_step0_step1_report.sh
│   ├── run_step2_step3_report.sh
│   └── run_step4_report.sh
├── reports/                          # 报告下载、统计脚本及本地 PDF/图表
├── oralgpt_cmf_llamafactory_sft_dataset.json   # 原始患者数据集（SFT 格式）
├── SH9HCMFdata/                      # 影像与表格原始数据（png/xlsx/jpg）
├── outputs/group<N>/<NAME>/          # 每个病人独立的流水线产物
│
├── step1_patient_trajectory/         # Step1：阶段切分与轨迹生成
│   ├── dataset.py                    #   加载患者、拆分问答轮次、对齐图片
│   ├── stages.py                     #   按临床释放顺序切分阶段、划分 held-out
│   ├── trajectories.py               #   标准轨迹 + 缺失模态变体 + 长噪声变体
│   └── noise_pool.json               #   冻结的长噪声池（按 patient_id 确定性采样）
│
├── step2_evidence/                   # Step2：原子证据与证据图
│   ├── pipeline.py                   #   单病人核心流程（轨迹、证据、证据图）
│   ├── run_step1_step2.py            #   多病人 Step1/2 入口
│   ├── evidence.py                   #   调用 LLM 并行抽取各阶段原子证据（带缓存）
│   ├── graph.py                      #   规则化构建证据图（节点 / 边）
│   ├── visualize_graph.py            #   渲染证据图 HTML 与 PNG
│   └── prompts/
│       ├── evidence_extraction.yaml  #   证据抽取 prompt 模板
│       └── graph_edges.yaml          #   跨阶段证据图边生成 prompt 模板
│
└── step3_tasks/                      # Step3：Benchmark 任务与评分 rubric
    ├── selectors.py                  #   证据索引/引用、规范化答案、任务规格组装
    ├── llm_tasks.py                  #   LLM 任务规划 + 问题生成 + 校验 + held-out 证据归因 + rubric 生成（带缓存）
    ├── run_step3.py                  #   并行多病人任务与 rubric 生成入口
    └── prompts/
        ├── normal_task_plan.yaml     #   普通任务的类型/数量/定义配置
        ├── task_planning.yaml        #   普通任务规划 prompt 模板（按任务类型逐类调用）
        ├── question_generation.yaml  #   问题生成 prompt 模板
        ├── qa_validation.yaml        #   问题/答案校验 prompt 模板
        ├── evidence_selection.yaml   #   held-out QA 证据归因 prompt 模板
        └── rubric_generation.yaml    #   诊断/治疗任务评分 rubric 生成 prompt 模板

└── step4_evaluation/                 # Step4 + Step5：记忆方法评测与打分
    ├── run_step4.py                  #   并行多病人流式测评与打分入口
    ├── evaluator.py                  #   流式评测引擎（缓存 LLM、逐阶段读取轨迹、多模态图片编码）
    ├── report.py                     #   汇总 ACC / ERS / 诊断分 / TPS，多方法对比表
    ├── scoring.py                    #   base 任务 ACC 判定与 rubric 打分（调用 LLM 裁判）
    ├── templating.py                 #   prompt 模板渲染
    ├── memory/                       #   记忆方法（一种方法一个文件）
    │   ├── base.py                   #     基类 MemoryMethod + 共享工具 format_stage_input / collect_stage_images
    │   ├── single_stage_memory.py    #     单阶段基线：每阶段清空，只保留当前阶段
    │   ├── full_context_memory.py    #     全上下文基线：拼接全部历史阶段原文
    │   ├── summary_memory.py         #     记忆基线：LLM 增量把每阶段融入一份紧凑记忆
    │   └── mem0_memory.py            #     mem0 检索式记忆（可选，需 pip install mem0ai）
    └── prompts/
        ├── answer.yaml               #   基于记忆回答问题 prompt 模板
        ├── memory_update.yaml        #   summary_memory 增量巩固 prompt 模板
        ├── judge_base.yaml           #   base 任务 ACC 判定与证据覆盖统计 prompt 模板
        └── judge_rubric.yaml         #   诊断/治疗 rubric 打分 prompt 模板
```

---

## 二、环境与配置

### 依赖

```bash
conda create -n cmfbench python=3.10 -y
conda activate cmfbench
pip install -r requirement.txt
playwright install chromium
```


### 配置项（`.env`）

`config.py` 通过 `load_env()` 从项目根目录的 `.env` 读取环境变量，再由 `get_settings()` 组装成 `Settings`：

```bash
# 通用默认配置
OPENAI_API_KEY=你的默认key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3.6-chat

# 生成 benchmark 的模型
BENCHMARK_OPENAI_API_KEY=你的生成模型key
BENCHMARK_OPENAI_BASE_URL=https://api.openai.com/v1
BENCHMARK_OPENAI_MODEL=qwen3.6-chat

# 回答问题 / 被测模型
ANSWER_OPENAI_API_KEY=你的回答模型key
ANSWER_OPENAI_BASE_URL=https://api.openai.com/v1
ANSWER_OPENAI_MODEL=gpt-4o-mini

# verifier / judge / critic 模型
VERIFIER_OPENAI_API_KEY=你的校验模型key
VERIFIER_OPENAI_BASE_URL=https://api.openai.com/v1
VERIFIER_OPENAI_MODEL=gpt-4o

# mem0 embedding
EMBEDDING_OPENAI_API_KEY=你的embedding模型key
EMBEDDING_OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EOF
```

固定数据路径：

- 数据集：`oralgpt_cmf_llamafactory_sft_dataset.json`
- 原始数据：`SH9HCMFdata/group1` 至 `SH9HCMFdata/group9`
- 病人输出：`outputs/<group>/<patient_name>`

---

## 三、批量运行

三个阶段分开启动。`--num-workers` 控制病人级并行；Step2、Step3 和 Step4 还分别支持阶段、任务和评估内部并行。总请求压力会叠加，本地单实例 VLM 建议保持 `--num-workers 1`，再按下面的内部并发参数逐步调高。所有脚本在 WSL 中自动激活 `cmfbench`。

患者范围：

- `--all`：运行数据集中的全部病人；
- `--limit N`：按数据集顺序只运行前 N 个病人。

默认会检查最终产物并跳过已完成病人；中断后重新执行相同命令即可继续，已有 LLM 缓存会复用。Step4 还会分别复用每个 memory method 的 `answers.json` 和 `report.json`。`--force` 会绕过完成判断；在 Step4 中会重新执行所选方法，但底层单请求缓存仍可命中。

### 1. 生成 Step1/2

```bash
# 前 4 个病人并行；每个病人最多并行抽取 2 个 stage
bash scripts/run_step1_step2.sh --limit 4 --num-workers 4 --stage-workers 2

# 全部病人；根据 benchmark API 吞吐量调整外层并发
bash scripts/run_step1_step2.sh --all --num-workers 4 --stage-workers 2
```

Step1 顺序生成标准轨迹及变体；Step2 使用 `--stage-workers` 并行抽取不同阶段的原子证据，随后构建证据图。输出仍按原阶段顺序汇总。

### 2. 生成 Step3 benchmark

Step1/2 完成后执行：

```bash
bash scripts/run_step3.sh --limit 4 --num-workers 2 --task-workers 4
bash scripts/run_step3.sh --all --num-workers 2 --task-workers 4
```

`--task-workers` 同时用于 evaluation QA 的证据选择和 treatment/followup rubric 生成；任务规划中的多轮生成、校验和反馈仍保持串行。

### 3. 运行 Step4 测评

Step3 完成后执行。默认评估 `standard_trajectory` 和 `full_context_memory`：

```bash
bash scripts/run_step4.sh --limit 4 --num-workers 1 \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

指定多条轨迹、多个 memory method 或多模态模式：

```bash
bash scripts/run_step4.sh --all --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods single_stage_memory,full_context_memory,summary_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1

bash scripts/run_step4.sh --all --num-workers 1 --multimodal \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods full_context_memory,summary_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

`model_perception_trajectory` 必须先按被测模型生成，路径为 `trajectories/model_perception_trajectory/<answer_model>/model_perception_trajectory.json`。其他轨迹从 `trajectories/<name>/<name>.json` 读取。`--method-workers` 默认 `1`；只有多 GPU 或多服务实例时才建议设置为大于 `1`。Step4 会用共享并发额度限制跨方法的 Answer 总并发和 Verifier 总并发。

### 三个本地 VLM 的评测流程

服务器已提供三个 OpenAI-compatible 本地 VLM 服务：

| 模型 | 服务脚本 | 默认地址 | API model id |
| --- | --- | --- | --- |
| LLaVA-Med-7B | `/root/autodl-tmp/serve/start_llava_med_api.sh` | `http://127.0.0.1:8001/v1` | `llava-med-7b` |
| MedGemma | `/root/autodl-tmp/serve/start_medgemma_api.sh` | `http://127.0.0.1:8002/v1` | `medgemma` |
| OralGPT-Omni-7B | `/root/autodl-tmp/serve/start_oralgpt_omni_api.sh` | `http://127.0.0.1:8003/v1` | `oralgpt-omni-7b` |

三个模型都使用同一套 `step4_evaluation`、同一 verifier、同一轨迹和同一 memo 方法；每次只改变 `--answer-base-url` 和 `--answer-model`。先启动需要评测的模型服务：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cmfbench
bash /root/autodl-tmp/serve/start_llava_med_api.sh
# 或：bash /root/autodl-tmp/serve/start_medgemma_api.sh
# 或：bash /root/autodl-tmp/serve/start_oralgpt_omni_api.sh
```

先用一个病人做连通性检查：

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/models
```

模型感知轨迹也必须用同一个被测模型生成。下面以 LLaVA-Med 为例；另外两个模型只替换地址和 model id：

```bash
bash scripts/run_perception_trajectory.sh --limit 1 --num-workers 1 \
  --model llava-med-7b --base-url http://127.0.0.1:8001/v1
```

批量生成前 4 个病人的模型感知轨迹：

```bash
bash scripts/run_perception_trajectory.sh --limit 4 --num-workers 1 \
  --model llava-med-7b --base-url http://127.0.0.1:8001/v1
```

然后运行标准轨迹和模型感知轨迹：

```bash
bash scripts/run_step4.sh \
  --limit 4 --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods full_context_memory,summary_memory \
  --answer-model llava-med-7b \
  --answer-base-url http://127.0.0.1:8001/v1 \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

MedGemma 和 OralGPT-Omni 分别使用：

```bash
bash scripts/run_step4.sh --limit 4 --num-workers 1 --trajectories standard_trajectory,model_perception_trajectory --methods full_context_memory,summary_memory --answer-model medgemma --answer-base-url http://127.0.0.1:8002/v1 --answer-workers 2 --score-workers 1 --method-workers 1

bash scripts/run_step4.sh --limit 4 --num-workers 1 --trajectories standard_trajectory,model_perception_trajectory --methods full_context_memory,summary_memory --answer-model oralgpt-omni-7b --answer-base-url http://127.0.0.1:8003/v1 --answer-workers 2 --score-workers 1 --method-workers 1
```

建议先以 `--num-workers 1 --answer-workers 1 --method-workers 1` 做单模型冒烟测试，再将 `--answer-workers` 提高到 `2`。MedGemma / OralGPT-Omni-7B 的长 prompt 和 `summary_memory` 较慢，不建议同时提高病人级与 method 级并发；`--score-workers` 主要作用于独立 verifier 服务。

### 产物

| 步骤 | 产物 |
| --- | --- |
| Step1 | `trajectories/standard_trajectory.json`、`trajectories/<变体名>/<变体名>.json` |
| Step2 | `evidence/evidence.json`、`graph/evidence_graph.json`、`graph/evidence_graph.html`、`graph/evidence_graph.png`、`cache/...` |
| Step3 | `tasks/all_tasks.json`、按任务类型分组的 JSON、`rubrics/treatment_rubrics.json`、`cache/step3/...` |
| Step4 | 每个方法的 `evaluation/<轨迹>/<answer_model>/<方法>/<text|multimodal>/answers.json` 和 `report.json`；多方法汇总位于 `evaluation/<轨迹>/<answer_model>/<text|multimodal>/report.json`、`report.txt`；缓存位于 `cache/step4/<轨迹>/<answer_model>/...` |

---

## 四、临床阶段

### 阶段定义

| 阶段 | 类型 | 模态 | 源轮次 |
| --- | --- | --- | --- |
| `S0_PROFILE` | 基本信息/文本 | TEXT_QA | 1,2,3 |
| `S1_FP` | 面像照片 | FP | 4 |
| `S2_DP` | 口内照片 | DP | 5 |
| `S3_XR_XLA` | 头影测量 + 全景片 | XR, XLData | 7,9 |
| `S4_CT` | 三维 CT | CT | 8 |
| `S5_TMJ` | 颞下颌关节 | TMJ | 6 |
| `S6_TREATMENT` | 诊断与治疗 evaluation（金标准） | TEXT_QA | 10–18 |


### 任务类型

| 任务类型 | 数量 | 考察能力 |
| --- | --- | --- |
| `modality_perception` | 3 | 当前模态感知 |
| `longitudinal_evidence_recall` | 2 | 纵向证据回忆 |
| `cross_modal_reasoning` | 2 | 跨模态整合推理 |
| `memory_update_conflict_correction` | 2 | 记忆更新 / 冲突纠正 |
| `treatment` | 9 | 诊断与治疗 evaluation（Q10–Q18，每题一个任务，原始 QA 直用，不改写） |



---

## 五、Step4 评测

对同一条临床轨迹，按阶段流式读取信息，并在每个阶段结束后释放并回答该阶段的任务；再对作答打分，汇总不同记忆方法的对比报告。

### 运行

使用 `scripts/run_step4.sh` 独立启动测评：

```bash
# 默认：前 4 个病人的标准轨迹 + full_context_memory
bash scripts/run_step4.sh --limit 4 --num-workers 1 \
  --answer-workers 2 --score-workers 1 --method-workers 1

# 指定轨迹变体和多种记忆方法
bash scripts/run_step4.sh --all --num-workers 1 \
  --trajectories long_noisy,no_ct \
  --methods single_stage_memory,summary_memory,mem0_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1

# 多模态输入
bash scripts/run_step4.sh --all --num-workers 1 --multimodal \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

### 命令行参数

| 参数 | 缺省 | 说明 |
| --- | --- | --- |
| `--all` / `--limit N` | 必选 | 全部病人，或数据集中的前 N 个病人 |
| `--num-workers` | `1` | 并行处理的病人数 |
| `--trajectories` | `standard_trajectory` | 逗号分隔的完整轨迹名；模型感知轨迹使用 `model_perception_trajectory` |
| `--methods` | `full_context_memory` | 逗号分隔的记忆方法 |
| `--multimodal` | 关闭 | 开启多模态图片输入 |
| `--answer-workers` | `2` | 同一轨迹内并行回答数，只允许 `1` 或 `2`；跨 method 共享此上限 |
| `--score-workers` | `1` | verifier 并行评分数，允许 `1`–`4`；跨 method 共享此上限。默认单路评分以获得最高稳定性 |
| `--method-workers` | `1` | 并行 memory method 数；仅多 GPU 或多服务实例时建议大于 `1` |
| `--force` | 关闭 | 重新执行所选方法和评分；底层单请求缓存仍可复用 |
| `--answer-model` | `.env` 的 `ANSWER_OPENAI_MODEL` | 覆盖本次被测回答模型名，并用于结果目录隔离 |
| `--answer-base-url` | `.env` 的 `ANSWER_OPENAI_BASE_URL` | 覆盖本次被测模型的 OpenAI-compatible 地址 |

同一阶段的问题会并行回答，阶段之间及 `summary_memory` 更新仍按时间顺序执行。评分按任务并行。Answer 与 Verifier 使用不同服务时，默认还能在方法之间形成回答/评分流水线；`--method-workers > 1` 则会直接并行运行不同 memory method。

当前 Step4 单次请求超时为 300 秒。输出上限分别为：treatment 回答 4096、其他回答 2048、`summary_memory` 4096、base/rubric/evidence 评分 2048 tokens。单个评分请求连续四次失败时，该题会记录到方法报告的 `failed_tasks` 并跳过，其他题和患者继续运行；含失败任务的报告不会被视为完成，重新执行同一命令会复用成功缓存并补评失败题。

### 记忆方法（`step4_evaluation/memory/`）

| `--methods` 取值 | 类 | 说明 |
| --- | --- | --- |
| `single_stage_memory` | `SingleStageMemory` | 单阶段基线：每阶段清空，只保留当前阶段（用于验证"仅当前阶段无法完成跨阶段任务"） |
| `full_context_memory` | `FullContextMemory` | 全上下文基线：拼接全部历史阶段原文（长上下文） |
| `summary_memory` | `SummaryMemory` | 记忆基线：LLM 增量把每阶段融入一份紧凑结构化记忆 |
| `mem0_memory` | `Mem0Memory` | mem0 检索式记忆：抽取事实入向量库、按问题语义检索（需 `pip install mem0ai`） |

> **扩展新方法**：在 `memory/` 下新建 `<name>.py` 继承 `MemoryMethod`（实现 `reset/observe/context`，需巩固再实现 `update`，需落盘再重写 `setup(workdir)`），并在 `memory/__init__.py` 的 `_REGISTRY` 登记即可被 `--methods` 选中，通用 pipeline 无需改动。

### 评分指标（`report.py`）

| 指标 | 适用任务 | 说明 |
| --- | --- | --- |
| **ACC** | base 任务（感知/纵向证据/跨模态/记忆更新） | LLM 裁判二元判定准确率；另按任务类型、模态细分 |
| **ERS** | 全部任务（感知/纵向证据/跨模态/记忆更新/治疗） | benchmark 生成阶段预先筛选的 `selected_evidence` 中，被模型答案正确覆盖的证据数 / 证据总数；所有任务统一口径，另按任务类型、模态细分 |
| **TPS** | `treatment` | 所有诊断与治疗 evaluation 的 rubric 得分均值（百分比） |



### 产物（`outputs/.../evaluation/<轨迹>/<answer_model>/`）

- `<方法>/<text|multimodal>/answers.json`：该模型、轨迹和 memory method 的逐任务作答记录
- `<方法>/<text|multimodal>/report.json`：单个 method 的评分检查点，用于中断续跑
- `<text|multimodal>/report.json` / `report.txt`：同一模型下的多方法对比报告
- `report.json` 同时记录 `answer_model`、`answer_base_url`、`verifier_model`、`verifier_base_url` 和 `memory_methods`

> mem0 的向量库持久化在 `cache/step4/<轨迹>/<answer_model>/mem0_memory/<text|multimodal>/vector_store/`，由方法自身经 `setup()` 配置；嵌入模型由 `.env` 的 `EMBEDDING_MODEL`（默认 `text-embedding-3-small`）指定。embedding key/url 优先使用 `EMBEDDING_OPENAI_*`，否则回退到通用 `OPENAI_*`；若未配置通用 key，必须配置 `EMBEDDING_OPENAI_API_KEY`。



评测结果按“轨迹 × 被测回答模型 × memo 方法 × 输入模式”比较。verifier/judge、benchmark 任务、rubric、随机种子和 generation 参数固定，只改变正在研究的因素。

感知误差：在同一个 `answer_model` 和 memo 方法下比较 `standard` 与 `model_perception`。两者的差值表示感知误差传播到记忆和后续任务后的影响；同时报告 `trajectories/model_perception_trajectory/<answer_model>/perception_report.json` 的 precision、recall、F1 和 hallucination control。

memo/检索误差：只比较不同 memo 方法能回答端到端效果，但不能单独证明下降来自检索，因为方法同时改变了记忆表示、更新/压缩和上下文长度。建议固定 `answer_model`，同时报告 `full_context_memory`、`summary_memory`、`mem0_memory`，并审计每道题 `answers.json` 里的 `memory_context` 与 `selected_evidence`：

```text
retrieval loss 上界 = score(full_context) - score(actual_memo)
```

这项只能称为记忆/检索瓶颈的上界估计。若 selected evidence 没出现在 `memory_context`，属于可解释的检索/记忆覆盖失败；若已出现但答案仍错，才进入推理错误候选。

推理误差：增加 oracle-context 条件，把该题 `selected_evidence` 对应的 gold facts 或人工核验后的完整相关证据直接注入 answer model，不经过感知、压缩和检索。固定同一个 answer model 比较 `oracle-context` 与实际 memo：oracle-context 仍答错的部分是回答模型的推理/表达失败；oracle-context 正确而实际 memo 错的部分属于记忆/检索链路损失。oracle 结果是诊断上界，不替代主结果。

当前轨迹变体可支持：`standard` vs `model_perception`（感知传播）、单缺失 vs 双缺失模态（模态协同）、`long_noisy`（抗噪和幻觉）、text vs multimodal（视觉输入收益）、不同历史长度/阶段位置（长上下文退化和 recency bias），以及 full-context/summary/mem0 的交叉比较。报告时按 overall、任务类型、模态和轨迹严重度分层，并按病人报告均值、标准差和 bootstrap 置信区间；单病人或极少任务不应写成稳定统计结论。

## 六、Report 长程病例流水线（`report_pipeline/`）

在原有「单次就诊、按**模态**切分 S0–S5」数据之外，本流水线从 `reports/` 下的**文献长程病例报告 PDF** 自动构造与 `oralgpt_cmf_llamafactory_sft_dataset.json` **同构**的数据，但阶段轴改为**时间**（就诊/随访时间点，跨月至跨年），用于扩充数据来源并新增「跨时间点记忆 / 趋势追踪 / 治疗-结局」这一评测维度。

### 与原数据的关系

| | 原数据（group1 病人） | 本流水线（report） |
| --- | --- | --- |
| 阶段轴 | 模态（S0_PROFILE … S5_TMJ） | **时间**（如 `T0_perception_00`、`T1_treatment_00`，含 `timepoint.t_months`） |
| 来源 | SH9H 真实病例 | 开放获取文献病例报告 PDF |
| `group` | `group1` 等 | `report`（产物隔离在 `outputs/report/<name>/`） |
| 产物格式 | `conversations` / 标准轨迹 | `dataset_entry.json` / 标准轨迹（下游 step2/3/4 可复用） |

### 全自动流程（LLM 抽取 ↔ 校验模型反馈循环）

```
step0 摄取(确定性)          抽取 <-> 校验 反馈循环                 step1(确定性)
PDF ─MinerU─► 全文/表格/图片 ─►  抽取模型 extract_timeline ──►         ──► SFT 条目
        图注↔图片语义配对       ▲                     │                  时间点阶段
                              └── 反馈(问题) ◄── 校验模型 verify ──┘      标准轨迹
                                   (对照原文+表格+图注核验)
```

- **step0 摄取**：用 **MinerU** 解析 PDF，产出逐页全文（`fulltext.json`）、**结构化表格 HTML**（`tables.json`，表格即图片也能识别为表）、图片，并用 MinerU 的**图注↔图片语义配对**（以 `Figure N` 为权威身份）生成 `captions.json`。
- **抽取模型**：从病例正文+表格+图注中直接抽取按时间点组织的 `qa_pairs`；以 `stage_type=perception|treatment|followup` 和 `role=observation|evaluation` 区分可见事实与隐藏评测答案，数量由论文内容决定。
- **校验模型**：对照原文+表格+图注核验事实、问题、角色、阶段、图片 QA、最早事件覆盖、数值与时序；high/medium 问题会作为**反馈**回灌给抽取模型，最多 `--max-iters` 轮。
- **step1**：校验 schema 和时序、解析图片、按时间点切分阶段，生成 `dataset_entry.json` 和唯一阶段数据源 `standard_trajectory.json`。


### 目录结构

```
report_pipeline/
├── step0_ingest/                      # PDF 摄取、时间线抽取与校验
├── step1_report_trajectory/           # 报告时间点阶段化与标准轨迹
├── run_step0_step1_report.py          # PDF -> 标准轨迹
├── run_step2_step3_report.py          # 标准轨迹 -> evidence/tasks/rubrics
└── run_step4_report.py                # 独立评估已生成的 benchmark
```

### 运行

```bash
# 运行 reports/pdf/ 下全部报告
bash scripts/run_step0_step1_report.sh --all --num-workers 2

bash scripts/run_step2_step3_report.sh --all --num-workers 2 \
  --stage-workers 2 --task-workers 4

bash scripts/run_step4_report.sh --all --num-workers 1 \
  --methods full_context_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1

# 先用一篇报告做测试
bash scripts/run_step0_step1_report.sh --limit 1 --num-workers 1

bash scripts/run_step2_step3_report.sh --limit 1 --num-workers 1 \
  --stage-workers 2 --task-workers 4

bash scripts/run_step4_report.sh --limit 1 --num-workers 1 \
  --methods full_context_memory \
  --answer-workers 1 --score-workers 1 --method-workers 1
```

通用参数与病人侧一致：

| 参数 | 说明 |
| --- | --- |
| `--all` | 处理 `reports/pdf/` 下全部 PDF |
| `--limit N` | 处理按文件名排序后的前 N 篇报告 |
| `--num-workers N` | 并行处理的报告数，默认 1 |
| `--force` | 已有最终产物时仍重新运行 |

额外参数：

- Step0/1：`--max-iters`、`--model`；单篇报告内部的摄取、抽取/校验反馈循环和轨迹生成保持串行。
- Step2/3：`--stage-workers` 默认 `2`，`--task-workers` 默认 `4`。
- Step4：`--methods`、`--multimodal`、`--answer-model`、`--answer-base-url`、`--answer-workers`、`--score-workers`、`--method-workers`。

Step0 摄取、时间线抽取和 Step1 轨迹分别检查已有产物并自动续跑。Step4 支持 method 级答案和评分续跑。三个 report 脚本会自动激活 `cmfbench`，并将额外命令行参数传给对应 Python 入口。

### 产物（`outputs/report/<PDF stem>/`）

| 路径 | 说明 |
| --- | --- |
| `raw/`、`images/` | Step0 的 PDF 解析结果与图片 |
| `timeline.extracted.json`、`verification_report.json` | 时间线抽取和校验记录 |
| `trajectories/standard_trajectory.json`、`dataset_entry.json` | Step1 标准轨迹和 SFT 条目 |
| `evidence/evidence.json`、`graph/` | Step2 证据及证据图 |
| `tasks/`、`rubrics/` | Step3 benchmark 任务和评分 rubric |
| `evaluation/standard_trajectory/<answer_model>/` | Step4 评估根目录；各方法的答案和评分位于 `<方法>/<text|multimodal>/`，多方法汇总位于 `<text|multimodal>/` |
