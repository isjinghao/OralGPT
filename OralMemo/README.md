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
├── scripts/                          # WSL 批量运行入口
│   ├── run_step1_step2.sh
│   ├── run_step3.sh
│   └── run_step4.sh
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
│   ├── pipeline.py                   #   单病人串行核心流程（轨迹、证据、证据图）
│   ├── run_step1_step2.py            #   并行多病人 Step1/2 入口
│   ├── evidence.py                   #   调用 LLM 逐阶段抽取原子证据（带缓存）
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

三个阶段必须分开手动启动。单个病人内部始终串行，不同病人通过 `--num-workers` 并行。所有脚本在 WSL 中自动激活 `cmfbench`。

患者范围：

- `--all`：运行数据集中的全部病人；
- `--limit N`：按数据集顺序只运行前 N 个病人。

默认会检查最终产物并跳过已完成病人；中断后重新执行相同命令即可继续，已有 LLM 缓存会复用。`--force` 仅忽略病人级完成判断。

### 1. 生成 Step1/2

```bash
# 前 4 个病人，4 个病人并行
bash scripts/run_step1_step2.sh --limit 4 --num-workers 4

# 全部病人
bash scripts/run_step1_step2.sh --all --num-workers 8
```

Step1/2 对每个病人串行生成标准轨迹、缺失模态/长噪声变体、原子证据和证据图。

### 2. 生成 Step3 benchmark

Step1/2 全部完成后再手动执行：

```bash
bash scripts/run_step3.sh --limit 4 --num-workers 4
bash scripts/run_step3.sh --all --num-workers 8
```

### 3. 运行 Step4 测评

Step3 全部完成后再手动执行。默认是标准轨迹和 `full_context_memory`：

```bash
bash scripts/run_step4.sh --limit 4 --num-workers 4
bash scripts/run_step4.sh --all --num-workers 4
```

指定多条轨迹、多个 memo 方法或多模态模式：

```bash
bash scripts/run_step4.sh --all --num-workers 4 \
  --trajectories standard,model_perception \
  --methods single_stage_memory,full_context_memory,summary_memory

bash scripts/run_step4.sh --all --num-workers 3 \
  --trajectories standard,model_perception \
  --methods full_context_memory,summary_memory \
  --multimodal
```

`model_perception` 要求病人目录中已存在 `trajectories/model_perception_trajectory.json`。其他轨迹名从 `variants/<name>.json` 读取。`report.json` 和 `report.txt` 记录本次命令指定的方法；需要对比多个 memo 方法时，请在同一次命令中全部传给 `--methods`。

### 产物

| 步骤 | 产物 |
| --- | --- |
| Step1 | `trajectories/standard_trajectory.json`、`variants/*.json` |
| Step2 | `evidence/evidence.json`、`graph/evidence_graph.json`、`graph/evidence_graph.html`、`graph/evidence_graph.png`、`cache/...` |
| Step3 | `tasks/all_tasks.json`、按任务类型分组的 JSON、`rubrics/treatment_rubrics.json`、`cache/step3/...` |
| Step4 | `evaluation/<轨迹>[_mm]/answers_<方法>.json`、`report.json`、`report.txt`、`cache/step4/...` |

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

使用前文的 `scripts/run_step4.sh` 独立启动测评。示例：

```bash
# 默认：前 4 个病人的标准轨迹 + full_context_memory
bash scripts/run_step4.sh --limit 4 --num-workers 4

# 指定多条轨迹和记忆方法
bash scripts/run_step4.sh --all --num-workers 4 \
  --trajectories long_noisy,no_ct \
  --methods single_stage_memory,summary_memory,mem0_memory

# 多模态；缓存和输出使用 _mm 后缀
bash scripts/run_step4.sh --all --num-workers 3 --multimodal
```

### 命令行参数

| 参数 | 缺省 | 说明 |
| --- | --- | --- |
| `--all` / `--limit N` | 必选 | 全部病人，或数据集中的前 N 个病人 |
| `--num-workers` | `1` | 并行处理的病人数；单病人内部仍串行 |
| `--trajectories` | `standard` | 逗号分隔的轨迹名；支持 `standard`、`model_perception` 和 `variants/` 下文件名 |
| `--methods` | `full_context_memory` | 逗号分隔的记忆方法 |
| `--multimodal` | 关闭 | 开启多模态图片输入 |
| `--force` | 关闭 | 忽略病人级完成判断，继续复用细粒度缓存 |

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



### 产物（`outputs/.../evaluation/<轨迹>[_mm]/`）

- `answers_<方法>.json`：各方法逐任务的作答记录（含 `n_images`）
- `report.json` / `report.txt`：多方法对比报告（结构化 + 控制台表格）

> mem0 的向量库持久化在 `cache/step4/<轨迹>[_mm]/mem0_memory/vector_store/`，由方法自身经 `setup()` 配置；嵌入模型由 `.env` 的 `EMBEDDING_MODEL`（默认 `text-embedding-3-small`）指定。embedding key/url 优先使用 `EMBEDDING_OPENAI_*`，否则回退到通用 `OPENAI_*`；若未配置通用 key，必须配置 `EMBEDDING_OPENAI_API_KEY`。

---

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

- **step0 摄取**（不经 LLM，确定性、通用）：用 **MinerU（pipeline 后端）** 解析 PDF，产出逐页全文（`fulltext.json`）、**结构化表格 HTML**（`tables.json`，表格即图片也能识别为表）、图片，并用 MinerU 的**图注↔图片语义配对**（以 `Figure N` 为权威身份）生成 `captions.json`。
- **抽取模型**：从（通用头/尾裁剪后的）病例正文+表格+图注中直接抽取按时间点组织的 `qa_pairs`；以 `stage_type=perception|treatment|followup` 和 `role=observation|evaluation` 区分可见事实与隐藏评测答案，数量由论文内容决定。
- **校验模型（critic）**：对照原文+表格+图注核验事实、问题、角色、阶段、图片 QA、最早事件覆盖、数值与时序；high/medium 问题会作为**反馈**回灌给抽取模型，最多 `--max-iters` 轮。
- **step1**（确定性、通用）：校验 schema 和时序、解析图片、按时间点切分阶段，生成 `dataset_entry.json` 和唯一阶段数据源 `standard_trajectory.json`。

> 全流程**无任何针对单篇 report 的写死内容**，换 PDF 即可复用。

### 目录结构

```
report_pipeline/
├── step0_ingest/                      # PDF 摄取、时间线抽取与校验
├── step1_report_trajectory/           # 报告时间点阶段化与标准轨迹
├── run_step0_step1_report.py          # PDF -> 标准轨迹
├── run_step2_step3_report.py          # 标准轨迹 -> evidence/tasks/rubrics
└── run_step4_report.py                # 独立评估已生成的 benchmark
```

输入目录固定为 `reports/pdf/`。报告名称、输出目录和 `patient_id` 自动取自 PDF 文件名：例如 `CR0001.pdf` 对应 `outputs/report/CR0001/` 和 `report__CR0001`，不再需要手动传入 `--name`。

### 运行

```bash
conda activate cmfbench

# 运行 reports/pdf/ 下全部报告
bash scripts/run_step0_step1_report.sh --all
bash scripts/run_step2_step3_report.sh --all
bash scripts/run_step4_report.sh --all

# 前 4 篇报告并行运行；单篇报告内部的 Step0 和 Step1 保持串行
bash scripts/run_step0_step1_report.sh --limit 4 --num-workers 4
bash scripts/run_step2_step3_report.sh --limit 4 --num-workers 4
bash scripts/run_step4_report.sh --limit 4 --num-workers 4
```

也可以直接运行 Python 入口：

```bash
python -m report_pipeline.run_step0_step1_report --all --num-workers 1
python -m report_pipeline.run_step2_step3_report --all --num-workers 1
python -m report_pipeline.run_step4_report --all --num-workers 1 --methods full_context_memory
```

通用参数与病人侧一致：

| 参数 | 说明 |
| --- | --- |
| `--all` | 处理 `reports/pdf/` 下全部 PDF |
| `--limit N` | 处理按文件名排序后的前 N 篇报告 |
| `--num-workers N` | 并行处理的报告数，默认 1 |
| `--force` | 已有最终产物时仍重新运行 |

Step0-1 额外支持 `--max-iters` 和 `--model`；Step4 额外支持 `--methods` 与 `--multimodal`。Step0 摄取、时间线抽取和 Step1 轨迹分别检查已有产物并自动续跑，`--force` 才会从头重跑。三个 Bash 脚本会自动激活 `cmfbench`，并将命令行参数原样传给对应 Python 入口。

### 产物（`outputs/report/<PDF stem>/`）

| 路径 | 说明 |
| --- | --- |
| `raw/`、`images/` | Step0 的 PDF 解析结果与图片 |
| `timeline.extracted.json`、`verification_report.json` | 时间线抽取和校验记录 |
| `trajectories/standard_trajectory.json`、`dataset_entry.json` | Step1 标准轨迹和 SFT 条目 |
| `evidence/evidence.json`、`graph/` | Step2 证据及证据图 |
| `tasks/`、`rubrics/` | Step3 benchmark 任务和评分 rubric |
| `evaluation/standard_full[_mm]/` | Step4 文本或多模态评估结果 |
