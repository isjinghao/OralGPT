# OralMemBench

面向口腔颌面外科（OralCMF）的多模态Benchmark。以患者多轮多模态问诊数据为输入，产出：

1. **分阶段临床轨迹**（按真实就诊顺序释放各模态证据）；
2. **原子证据**（用 LLM 从可见问答中抽取的最小事实单元）；
3. **证据图**（阶段内 / 跨阶段的依赖关系，含可视化 HTML）；
4. **Benchmark 任务**（感知、回忆、跨模态推理、记忆更新、诊断、治疗）及**评分 rubric**。

---

## 一、目录结构

```
bench/
├── __init__.py                       # 包说明
├── config.py                         # 配置：路径 + .env / OpenAI 设置
├── llm_client.py                     # OpenAI 兼容客户端（限流重试 + JSON 解析）
├── oralgpt_cmf_llamafactory_sft_dataset.json   # 原始患者数据集（SFT 格式）
├── SH9HCMFdata/                      # 影像与表格原始数据（png/xlsx/jpg）
├── outputs/group<N>/<NAME>/          # 每个病人独立的流水线产物（stages/trajectories/variants/evidence/cache 等），路径与原始数据 group<N>__<NAME> 命名一致
│
├── step1_patient_trajectory/         # Step1：阶段切分与轨迹生成
│   ├── dataset.py                    #   加载患者、拆分问答轮次、对齐图片
│   ├── stages.py                     #   按临床释放顺序切分阶段、划分 held-out
│   ├── trajectories.py               #   标准轨迹 + 缺失模态变体 + 长噪声变体
│   └── noise_pool.json               #   冻结的长噪声池（按 patient_id 确定性采样）
│
├── step2_evidence/                   # Step2：原子证据与证据图
│   ├── pipeline.py                   #   单病人核心流程（Step1 轨迹 + Step2 证据），供两个入口复用
│   ├── run_one.py                    #   入口①a：处理单个病人（可单独重跑某个失败病人）
│   ├── run_all.py                    #   入口①b：批量处理全部病人（遇错继续 + 进度 + 报告）
│   ├── evidence.py                   #   调用 LLM 逐阶段抽取原子证据（带缓存）
│   ├── graph.py                      #   规则化构建证据图（节点 / 边）
│   ├── visualize_graph.py            #   入口②：渲染证据图 HTML
│   └── prompts/
│       ├── evidence_extraction.yaml  #   证据抽取 prompt 模板
│       └── graph_edges.yaml          #   跨阶段证据图边生成 prompt 模板
│
└── step3_tasks/                      # Step3：Benchmark 任务与评分 rubric
    ├── selectors.py                  #   证据索引/引用、规范化答案、任务规格组装
    ├── llm_tasks.py                  #   LLM 任务规划 + 问题生成 + 校验 + held-out 证据归因 + rubric 生成（带缓存）
    ├── run_step3_chenfang.py         #   入口③：生成全部任务与 rubric
    └── prompts/
        ├── normal_task_plan.yaml     #   普通任务的类型/数量/定义配置
        ├── task_planning.yaml        #   普通任务规划 prompt 模板（按任务类型逐类调用）
        ├── question_generation.yaml  #   问题生成 prompt 模板
        ├── qa_validation.yaml        #   问题/答案校验 prompt 模板
        ├── evidence_selection.yaml   #   held-out QA 证据归因 prompt 模板
        └── rubric_generation.yaml    #   诊断/治疗任务评分 rubric 生成 prompt 模板

└── step4_evaluation/                 # Step4 + Step5：记忆方法评测与打分
    ├── run_step4_chenfang.py         #   入口④：按阶段流式提问作答并打分，汇总对比报告
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
```

> 代码需 **Python ≥ 3.10**。`requirement.txt` 已含全部依赖（其中 `mem0ai` 仅 Step4 `--methods mem0_memory` 时用到）。

### 配置项（`.env`）

`config.py` 通过 `load_env()` 从项目根目录的 `.env` 读取环境变量，再由 `get_settings()` 组装成 `Settings`：

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | 是 | 无（缺失会报 `KeyError`） | OpenAI 兼容接口的 API Key |
| `OPENAI_BASE_URL` | 否 | `https://api.openai.com/v1` | 接口基址（末尾 `/` 会被去除） |
| `OPENAI_MODEL` | 否 | `qwen3.6-chat` | 使用的模型名 |
| `EMBEDDING_MODEL` | 否 | `text-embedding-3-small` | 仅 Step4 `mem0_memory` 用；向量化模型，复用 `OPENAI_API_KEY` / `OPENAI_BASE_URL` |

在项目根目录创建 `.env`：

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3.6-chat
EMBEDDING_MODEL=text-embedding-3-small
EOF
```

`Settings` 中固定的路径（无需配置）：

- `dataset_json` = `bench/oralgpt_cmf_llamafactory_sft_dataset.json`
- `data_root`    = `bench/SH9HCMFdata`
- `output_root`  = `bench/outputs/group1/CHENFANG`

---

## 三、运行方式

```bash
conda activate cmfbench

# Step1 + Step2
python -m bench.step2_evidence.run_all                  # 批量处理全部病人，默认断点续跑
python -m bench.step2_evidence.run_all --force          # 忽略已有结果，强制重跑全部
python -m bench.step2_evidence.run_one group1__CHENFANG # 仅处理单个病人

# Step2 可视化
python -m bench.step2_evidence.visualize_graph

# Step3：生成任务与评分
python -m bench.step3_tasks.run_step3_chenfang

# Step4：记忆方法评测与打分（详见「五、Step4 评测」）
python step4_evaluation/run_step4_chenfang.py
```

### 产物

| 步骤 | 产物 |
| --- | --- |
| Step1 | `stages/patient_stages.json`、`trajectories/standard_trajectory.json`、`variants/*.json` |
| Step2 | `evidence/evidence.json`、`cache/evidence_*.json` |
| Step2 可视化 | `graph/evidence_graph.json`、`graph/evidence_graph.html` |
| Step3 | `tasks/all_tasks.json` 及按类型分组的 6 个 json、`rubrics/{diagnosis,treatment}_rubrics.json`、`cache/step3/...` |
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
| held-out | 诊断 / 治疗（金标准） | — | 10（诊断），11–18（治疗） |

### 任务类型

| 任务类型 | 数量 | 考察能力 |
| --- | --- | --- |
| `modality_perception` | 3 | 当前模态感知 |
| `longitudinal_evidence_recall` | 2 | 纵向证据回忆 |
| `cross_modal_reasoning` | 2 | 跨模态整合推理 |
| `memory_update_conflict_correction` | 2 | 记忆更新 / 冲突纠正 |
| `heldout_diagnosis` | 1 | 综合诊断（held-out Q10，原始 QA 直用，不改写） |
| `heldout_treatment` | 8 | 治疗方案（held-out Q11–Q18，每题一个任务，原始 QA 直用，不改写） |

---

## 五、Step4 评测（记忆方法对比）

对同一条临床轨迹，按阶段**流式**读取信息（`observe` → `update`），并在每个阶段结束后释放并回答该阶段的任务；再对作答打分，汇总不同记忆方法的对比报告。缓存与输出均按 `trajectory_type`（及模态）隔离，互不污染。

### 运行

```bash
# 默认：标准轨迹 + single_stage_memory 一种方法
python step4_evaluation/run_step4_chenfang.py

# 指定多条轨迹（逗号分隔）
python step4_evaluation/run_step4_chenfang.py --trajectories long_noisy,no_ct

# 多模态：把记忆中的图片以 image_url 附给大模型（缓存/输出加 _mm 后缀，与纯文本互不污染）
python step4_evaluation/run_step4_chenfang.py --multimodal

# 指定要跑的记忆方法（逗号分隔，可多个）
python step4_evaluation/run_step4_chenfang.py --methods single_stage_memory,summary_memory,mem0_memory
```

### 命令行参数（均为可选，逗号分隔）

| 参数 | 缺省 | 说明 |
| --- | --- | --- |
| `--trajectories` | `standard` | 轨迹名：`standard` 或 `variants/` 下文件名（如 `long_noisy`、`no_ct`、`no_tmj`） |
| `--methods` | `single_stage_memory` | 要跑的记忆方法，取值 == `memory/` 下的文件名（也是类的 `name`） |
| `--multimodal` | 关闭 | 开启多模态图片输入 |

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
| **ERS** | base 任务（感知/纵向证据/跨模态/记忆更新） | `selected_evidence` 中被模型答案正确覆盖的证据数 / 证据总数；另按任务类型、模态细分 |
| **Diagnosis** | `heldout_diagnosis` | 按诊断 rubric 打分（百分比） |
| **TPS** | `heldout_treatment` | 各治疗任务 rubric 得分的均值（百分比） |

### 产物（`outputs/.../evaluation/<轨迹>[_mm]/`）

- `answers_<方法>.json`：各方法逐任务的作答记录（含 `n_images`）
- `report.json` / `report.txt`：多方法对比报告（结构化 + 控制台表格）

> mem0 的向量库持久化在 `cache/step4/<轨迹>[_mm]/mem0_memory/vector_store/`，由方法自身经 `setup()` 配置；嵌入模型由 `.env` 的 `EMBEDDING_MODEL`（默认 `text-embedding-3-small`）指定，复用 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。

---

## 六、Report 长程病例流水线（`report_pipeline/`）

在原有「单次就诊、按**模态**切分 S0–S5」数据之外，本流水线从 `reports/` 下的**文献长程病例报告 PDF** 自动构造与 `oralgpt_cmf_llamafactory_sft_dataset.json` **同构**的数据，但阶段轴改为**时间**（就诊/随访时间点，跨月至跨年），用于扩充数据来源并新增「跨时间点记忆 / 趋势追踪 / 治疗-结局」这一评测维度。

### 与原数据的关系

| | 原数据（group1 病人） | 本流水线（report） |
| --- | --- | --- |
| 阶段轴 | 模态（S0_PROFILE … S5_TMJ） | **时间**（`T0_… / T1_… / …`，含 `timepoint.t_months`） |
| 来源 | SH9H 真实病例 | 开放获取文献病例报告 PDF |
| `group` | `group1` 等 | `report`（物理隔离，独立 `report_dataset.json`） |
| 产物格式 | `conversations` / `stages` / `heldout` | **完全同构**（下游 step2/3/4 可复用） |

### 全自动流程（LLM 抽取 ↔ 校验模型反馈循环）

```
step0 摄取(确定性)          抽取 <-> 校验 反馈循环                 step1(确定性)
PDF ─MinerU─► 全文/表格/图片 ─►  抽取模型 extract_timeline ──►         ──► SFT 条目
        图注↔图片语义配对       ▲                     │                  时间点阶段
                              └── 反馈(问题) ◄── 校验模型 verify ──┘      标准轨迹
                                   (对照原文+表格+图注核验)
```

- **step0 摄取**（不经 LLM，确定性、通用）：用 **MinerU（pipeline 后端）** 解析 PDF，产出逐页全文（`fulltext.json`）、**结构化表格 HTML**（`tables.json`，表格即图片也能识别为表）、图片，并用 MinerU 的**图注↔图片语义配对**（以 `Figure N` 为权威身份）生成 `captions.json`。
- **抽取模型**：从（通用头/尾裁剪后的）病例正文+表格+图注中抽出结构化时间线（每个时间点的原子 findings、决策、依据 + held-out 诊断/治疗/预后），强约束「只用原文事实、数值/牙位/日期原样保留、表格与叙述冲突以表格为准」。
- **校验模型（critic）**：对照原文+表格+图注逐条核验事实支持性、数值保真、跨时间点逻辑一致与时序；若存在 high 级问题，则把问题清单作为**反馈**回灌给抽取模型自我修正，最多 `--max-iters` 轮。
- **step1**（确定性、通用）：把已校验的 findings 用通用模板渲染成问答（数值逐字），组装成 SFT 条目、按时间点切分阶段、复用 step1 的 `build_standard_trajectory` 生成标准轨迹，并追加进 `report_dataset.json`。

> 全流程**无任何针对单篇 report 的写死内容**，换 PDF 即可复用。

### 目录结构

```
report_pipeline/
├── config_reports.py                 # 复用 Settings + name 路径(产物统一在 outputs/report/<name>/)
├── step0_ingest/
│   ├── pdf_extract.py                 # MinerU 解析: 全文/表格(HTML)/图片 + 图注↔图片映射
│   ├── timeline_llm.py                # 抽取模型(反馈感知 + 头/尾裁剪)
│   ├── verify_llm.py                  # 校验模型(critic, 对照原文核验)
│   └── prompts/
│       ├── timeline_extraction.yaml   # 时间线抽取 prompt(通用)
│       └── timeline_verification.yaml # 校验 prompt(critic)
├── step1_report_trajectory/
│   ├── qa_render.py                   # 确定性: findings -> 有序问答轮次(通用模板)
│   ├── report_dataset.py              # -> 与原数据同构的 SFT 条目
│   ├── report_stages.py              # 按时间点切分阶段(替代按模态的 classify_turn)
│   └── report_trajectories.py         # 复用 step1 标准轨迹 + 回填 timepoint
└── run_report_pipeline.py             # 主编排(step0 + 反馈循环 + step1)
```

### 运行

```bash
conda activate cmfbench

# 处理一篇报告(PDF 路径 + name)
python report_pipeline/run_report_pipeline.py --pdf reports/s12903-026-09034-7_reference.pdf --name pls_8y --max-iters 3
```

| 参数 | 缺省 | 说明 |
| --- | --- | --- |
| `--pdf` | 必填 | 报告 PDF 路径（相对工作区根或绝对） |
| `--name` | 必填 | 报告标识，用于输出目录与病人 id（`report__<name>`） |
| `--max-iters` | 3 | 抽取↔校验反馈循环最大轮数 |
| `--model` | `.env` 的 `OPENAI_MODEL` | 覆盖使用的模型 |

### 产物（`outputs/report/<name>/`）

| 文件 | 说明 |
| --- | --- |
| `raw/{fulltext.json,tables.json,captions.json}` | step0 确定性抽取的中间产物（`captions.json` 仅存图注↔图片对齐表） |
| `images/*.jpeg` | 过滤去重后的内嵌图片 |
| `timeline.extracted.json` | LLM 抽取并经校验的结构化时间线（timepoints + held_out） |
| `verification_report.json` | 每轮校验记录（passed / issues / 反馈） |
| `stages/patient_stages.json` | 按时间点切分的阶段（含 `timepoint`） |
| `trajectories/standard_trajectory.json` | 标准轨迹 |
| `dataset_entry.json` | 该报告的 SFT 条目；并汇总进根目录 `report_dataset.json` |

### 已验证样本（通用性）

同一套代码/prompt 已跑通三篇结构迥异的纵向报告，均通过校验：

| name | 报告 | 时间点* | 说明 |
| --- | --- | --- | --- |
| `pls_8y` | Papillon–Lefèvre 8 年牙周维护 | ~6 | 有 CARE 时间线表(MinerU 解析为 HTML)；BOP 跨时间点可追踪 |
| `ph1_14m` | PH1 种植修复 14 个月 | ~5 | 用 ISQ 而非 BOP；术前→植入→6 月→14 月 |
| `pax7_dup` | PAX7 颅面重复畸形分期手术 | ~4 | 含产前/新生儿节点；ACMG 表(图片形式)被 MinerU 识别为表 |

> *时间点数量会因 LLM 抽取略有浮动，由 critic 每轮把关。

> **依赖**：`requirement.txt` 含 `mineru[core]`（含 torch + 版面/表格/OCR 模型；CPU 可跑，建议 GPU）。首次运行会联网下模型（默认 ModelScope 源）。
> **注意**：若所配置模型为推理型（思维链计入 `max_tokens`），本流水线已把抽取/校验的 `max_tokens` 提到 16000、`llm_client` 超时提到 300s；生产建议使用稳定支持 JSON 输出的模型。图注↔图片由 MinerU 按版面语义配对（以 `Figure N` 为身份），多面板密集图仍可能不完美。
