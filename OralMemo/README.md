# OralMemBench（`bench` 目录）

`bench` 目录是 **OralMemBench** 的构建流水线：面向口腔颌面外科（OralCMF）的多模态、纵向「记忆 + 推理」Benchmark。它以单个患者 `group1__CHENFANG` 的多轮多模态问诊数据为输入，自动产出：

1. **分阶段临床轨迹**（按真实就诊顺序释放各模态证据）；
2. **原子证据**（用 LLM 从可见问答中抽取的最小事实单元）；
3. **证据图**（阶段内 / 跨阶段的依赖关系，含可视化 HTML）；
4. **Benchmark 任务**（感知、回忆、跨模态推理、记忆更新、诊断、治疗）及**评分 rubric**。

> 设计原则：仅使用 Q1–Q9 的可见问答作为证据，Q10（诊断）与 Q11–Q18（治疗）作为「held-out 金标准答案」，不暴露给记忆构建过程。

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
```

---

## 二、环境与配置

### 依赖

```bash
conda create -n cmfbench python=3.10 -y
conda activate cmfbench
pip install openai PyYAML
```

> 代码需 **Python ≥ 3.10**。

### 配置项（`.env`）

`config.py` 通过 `load_env()` 从项目根目录的 `.env` 读取环境变量，再由 `get_settings()` 组装成 `Settings`：

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | 是 | 无（缺失会报 `KeyError`） | OpenAI 兼容接口的 API Key |
| `OPENAI_BASE_URL` | 否 | `https://api.openai.com/v1` | 接口基址（末尾 `/` 会被去除） |
| `OPENAI_MODEL` | 否 | `qwen3.6-chat` | 使用的模型名 |

在项目根目录创建 `.env`：

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3.6-chat
EOF
```

`Settings` 中固定的路径（无需配置）：

- `dataset_json` = `bench/oralgpt_cmf_llamafactory_sft_dataset.json`
- `data_root`    = `bench/SH9HCMFdata`
- `output_root`  = `bench/outputs/group1/CHENFANG`

---

## 三、运行方式（按顺序）

务必在**项目根目录**用模块方式运行（`bench` 内部用绝对导入 `from bench....`）：

```bash
conda activate cmfbench

# Step1 + Step2：阶段轨迹 + 原子证据
python -m bench.step2_evidence.run_all                 # 批量处理全部病人（遇错继续，末尾生成 outputs/_batch_report.json）
python -m bench.step2_evidence.run_all --skip-existing  # 断点续跑：跳过已完成的病人
python -m bench.step2_evidence.run_one group1__CHENFANG # 仅处理单个病人（用于单独重跑某个失败病人）

# Step2 可视化：构建证据图 JSON + HTML
python -m bench.step2_evidence.visualize_graph

# Step3：生成任务与评分 rubric
python -m bench.step3_tasks.run_step3_chenfang
```

### 产物路径（均在 `bench/outputs/group1/CHENFANG/` 下）

| 步骤 | 产物 |
| --- | --- |
| Step1 | `stages/patient_stages.json`、`trajectories/standard_trajectory.json`、`variants/*.json`（6 个变体） |
| Step2 | `evidence/evidence.json`、`cache/evidence_*.json`（分阶段缓存） |
| Step2 可视化 | `graph/evidence_graph.json`、`graph/evidence_graph.html` |
| Step3 | `tasks/all_tasks.json` 及按类型分组的 6 个 json、`rubrics/{diagnosis,treatment}_rubrics.json`、`cache/step3/...` |

---

## 四、临床阶段与任务体系

### 阶段定义（`stages.py` 中的 `STAGE_DEFS`）

| 阶段 | 类型 | 模态 | 源轮次 |
| --- | --- | --- | --- |
| `S0_PROFILE` | 基本信息/文本 | TEXT_QA | 1,2,3 |
| `S1_FP` | 面像照片 | FP | 4 |
| `S2_DP` | 口内照片 | DP | 5 |
| `S3_XR_XLA` | 头影测量 + 全景片 | XR, XLData | 7,9 |
| `S4_CT` | 三维 CT | CT | 8 |
| `S5_TMJ` | 颞下颌关节 | TMJ | 6 |
| held-out | 诊断 / 治疗（金标准） | — | 10（诊断），11–18（治疗） |

### 任务类型（普通任务配置见 `prompts/normal_task_plan.yaml`）

| 任务类型 | 数量 | 考察能力 |
| --- | --- | --- |
| `modality_perception` | 3 | 当前模态感知 |
| `longitudinal_evidence_recall` | 2 | 纵向证据回忆（含来源阶段） |
| `cross_modal_reasoning` | 2 | 跨模态整合推理 |
| `memory_update_conflict_correction` | 2 | 记忆更新 / 冲突纠正 |
| `heldout_diagnosis` | 1 | 综合诊断（held-out Q10，原始 QA 直用，不改写） |
| `heldout_treatment` | 8 | 治疗方案（held-out Q11–Q18，每题一个任务，原始 QA 直用，不改写） |

> **生成机制：**
> - **普通四类任务**：由 LLM 按 `normal_task_plan.yaml` 中配置的类型与数量**逐类规划**（每类一次调用，输出 `ask_after_stage`/`objective`/`required_evidence_ids`/`gold_answer`），证据 id 只能取自该病人的真实证据目录；再由 LLM 逐条生成问题并校验。任务类型与数量在 `normal_task_plan.yaml` 中调整。
> - **诊断与治疗任务（held-out QA）**：问题与金标准答案直接取自原始数据集 Q10–Q18，逐题拆分、不经 LLM 改写；其依赖的证据由 LLM 基于「问题 + 金标准 + 全证据目录 + 证据图」**归因**得到，子图边再由证据索引自动计算。
> - **评分 rubric**：由 LLM 基于每条诊断/治疗任务的问题、金标准与支撑证据生成，遍历全部诊断/治疗任务。
