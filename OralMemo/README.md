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
    ├── report.py                     #   汇总 ERS / 诊断分 / TPS，多方法对比表
    ├── scoring.py                    #   ERS 二元判定与 rubric 打分（调用 LLM 裁判）
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
        ├── judge_recall.yaml         #   召回类任务 ERS 二元判定 prompt 模板
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
| **ERS** | 召回类（感知/回忆/跨模态/记忆更新） | LLM 裁判二元判定正确率；另按任务类型、模态细分 |
| **Diagnosis** | `heldout_diagnosis` | 按诊断 rubric 打分（百分比） |
| **TPS** | `heldout_treatment` | 各治疗任务 rubric 得分的均值（百分比） |

### 产物（`outputs/.../evaluation/<轨迹>[_mm]/`）

- `answers_<方法>.json`：各方法逐任务的作答记录（含 `n_images`）
- `report.json` / `report.txt`：多方法对比报告（结构化 + 控制台表格）

> mem0 的向量库持久化在 `cache/step4/<轨迹>[_mm]/mem0_memory/vector_store/`，由方法自身经 `setup()` 配置；嵌入模型由 `.env` 的 `EMBEDDING_MODEL`（默认 `text-embedding-3-small`）指定，复用 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。
