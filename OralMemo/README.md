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
├── utils/                            # 共享批处理与 JSON 工具
├── scripts/                          # WSL 批量运行入口（自动激活 cmfbench）
│   ├── run_step1_step2.sh
│   ├── run_step3.sh
│   ├── run_step4.sh
│   ├── run_perception_trajectory.sh
│   ├── run_step0_step1_report.sh
│   ├── run_step2_step3_report.sh
│   ├── run_perception_trajectory_report.sh
│   └── run_step4_report.sh
├── reports/                          # 报告下载、统计脚本及本地 PDF/图表
├── oralgpt_cmf_llamafactory_sft_dataset.json   # 原始患者数据集（SFT 格式）
├── SH9HCMFdata/                      # 影像与表格原始数据（png/xlsx/jpg）
├── outputs/group<N>/<NAME>/          # 每个病人独立的流水线产物
│
├── step1_patient_trajectory/         # Step1：阶段切分与轨迹生成
│   ├── dataset.py                    #   加载患者、拆分问答轮次、对齐图片
│   ├── stages.py                     #   按临床释放顺序切分阶段、划分 held-out
│   ├── trajectories.py               #   标准轨迹 + 缺失模态变体 + 三档噪声变体
│   └── noise_pool.json               #   冻结噪声池（固定种子确定性采样）
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

# 可选：记忆构建模型；未设置时回退到 OPENAI_*
MEMO_OPENAI_API_KEY=你的记忆模型key
MEMO_OPENAI_BASE_URL=https://api.openai.com/v1
MEMO_OPENAI_MODEL=gpt-4o-mini

# 检索记忆共用 embedding
EMBEDDING_OPENAI_API_KEY=你的embedding模型key
EMBEDDING_OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Graphiti 使用 Neo4j
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=你的Neo4j密码
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

指定多条轨迹和多个 memory method：

```bash
bash scripts/run_step4.sh --all --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods single_stage_memory,full_context_memory,summary_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

`model_perception_trajectory` 必须先按被测模型生成，路径为 `trajectories/model_perception_trajectory/<answer_model>/model_perception_trajectory.json`。其他轨迹从 `trajectories/<name>/<name>.json` 读取。`--method-workers` 默认 `1`；只有多 GPU 或多服务实例时才建议设置为大于 `1`。Step4 会用共享并发额度限制跨方法的 Answer 总并发和 Verifier 总并发。

### 完整实验前的服务准备

完整实验是否需要启动服务，取决于所选轨迹、answer model 和 memory method：

| 功能 | 必需服务 / 配置 |
| --- | --- |
| Step1 / Step2 / Step3 | `.env` 中的 `BENCHMARK_OPENAI_*` |
| Step4 所有方法 | `.env` 中的 `ANSWER_OPENAI_*`、`VERIFIER_OPENAI_*` |
| `summary_memory`、`mem0_memory`、`langmem_memory`、`graphiti_memory` | `.env` 中的 `MEMO_OPENAI_*` |
| `vector_memory`、`mem0_memory`、`langmem_memory`、`graphiti_memory` | `.env` 中的 `EMBEDDING_OPENAI_*` |
| `graphiti_memory` | Neo4j 与 `.env` 中的 `GRAPHITI_NEO4J_*` |
| `model_perception_trajectory` 或本地 VLM 作为 answer model | 对应本地 VLM 的 OpenAI-compatible 服务 |

`single_stage_memory` 与 `full_context_memory` 不依赖 Memo LLM、embedding 或 Neo4j。运行完整 memory 对比前，先确认 `.env` 的 Memo、embedding 和 Neo4j 配置均已填写；不要把真实 key 或 Neo4j 密码提交到 Git。

### Autodl 容器：启动本地 VLM

当前 Autodl 容器的服务脚本位于 `/root/autodl-tmp/serve`。三个脚本会自行加载所需环境、后台运行、写入 PID/日志，并等待服务就绪；无需手动 `conda activate`：

| 模型 | 启动脚本 | 默认地址 | API model id | 实际运行环境 |
| --- | --- | --- | --- | --- |
| LLaVA-Med-7B | `start_llava_med_api.sh` | `http://127.0.0.1:8001/v1` | `llava-med-7b` | `cmfbench` |
| MedGemma | `start_medgemma_api.sh` | `http://127.0.0.1:8002/v1` | `medgemma` | `/root/autodl-tmp/venvs/medgemma` |
| OralGPT-Omni-7B | `start_oralgpt_omni_api.sh` | `http://127.0.0.1:8003/v1` | `oralgpt-omni-7b` | `/root/autodl-tmp/venvs/oralgpt-llamafactory`（LLaMA-Factory `qwen2_vl` 模板） |

按需启动一个或多个模型：

```bash
bash /root/autodl-tmp/serve/start_llava_med_api.sh
bash /root/autodl-tmp/serve/start_medgemma_api.sh
bash /root/autodl-tmp/serve/start_oralgpt_omni_api.sh
```

检查服务。LLaVA-Med 和 MedGemma 提供 `/health`；OralGPT 的官方 LLaMA-Factory API 使用 `/v1/models` 作为就绪检查：

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8003/v1/models

# 查看日志
tail -f /root/autodl-tmp/serve/logs/llava-med-api.log
# 或 medgemma-api.log / oralgpt-omni-api.log
```

停止全部本地模型服务：

```bash
bash /root/autodl-tmp/serve/stop_model_apis.sh
```

三个模型共用一张 GPU 时，建议每个模型先使用 `--num-workers 1 --answer-workers 1 --method-workers 1` 冒烟。不要同时提高病人级、answer 级和 method 级并发；`--score-workers` 访问的是独立的 verifier 服务，多个 Step4 进程共享同一个 verifier 时仍应控制总并发。

模型感知轨迹必须由同一个被测模型生成。以下以 LLaVA-Med 为例；MedGemma / OralGPT 只替换模型 ID 与地址：

```bash
bash scripts/run_perception_trajectory.sh --limit 1 --num-workers 1 \
  --model llava-med-7b --base-url http://127.0.0.1:8001/v1

bash scripts/run_step4.sh --limit 1 --num-workers 1 \
  --trajectories standard_trajectory,model_perception_trajectory \
  --methods full_context_memory,summary_memory,vector_memory,mem0_memory,langmem_memory,graphiti_memory \
  --answer-model llava-med-7b \
  --answer-base-url http://127.0.0.1:8001/v1 \
  --answer-workers 1 --score-workers 1 --method-workers 1
```

MedGemma 与 OralGPT-Omni 的参数：

```bash
# MedGemma
--answer-model medgemma --answer-base-url http://127.0.0.1:8002/v1

# OralGPT-Omni-7B
--answer-model oralgpt-omni-7b --answer-base-url http://127.0.0.1:8003/v1
```

### 产物

| 步骤 | 产物 |
| --- | --- |
| Step1 | `trajectories/standard_trajectory.json`、`trajectories/<变体名>/<变体名>.json` |
| Step2 | `evidence/evidence.json`、`graph/evidence_graph.json`、`graph/evidence_graph.html`、`graph/evidence_graph.png`、`cache/...` |
| Step3 | `tasks/all_tasks.json`、按任务类型分组的 JSON、`rubrics/treatment_rubrics.json`、`cache/step3/...` |
| Step4 | 每个方法的 `evaluation/<轨迹>/<answer_model>/<方法>/answers.json` 和 `report.json`；多方法汇总位于 `evaluation/<轨迹>/<answer_model>/report.json`、`report.csv`；缓存位于 `cache/step4/<轨迹>/<answer_model>/...` |

---

## 四、Step4 评测

对同一条临床轨迹，按阶段流式读取信息，并在每个阶段结束后释放并回答该阶段的任务；再对作答打分，汇总不同记忆方法的对比报告。

### 运行

使用 `scripts/run_step4.sh` 独立启动测评：

```bash
# 默认：前 4 个病人的标准轨迹 + full_context_memory
bash scripts/run_step4.sh --limit 4 --num-workers 1 \
  --answer-workers 2 --score-workers 1 --method-workers 1

# 指定轨迹变体和多种记忆方法
bash scripts/run_step4.sh --all --num-workers 1 \
  --trajectories short_noisy,medium_noisy,long_noisy,no_ct \
  --methods single_stage_memory,summary_memory,mem0_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1
```

### 命令行参数

| 参数 | 缺省 | 说明 |
| --- | --- | --- |
| `--all` / `--limit N` | 必选 | 全部病人，或数据集中的前 N 个病人 |
| `--num-workers` | `1` | 并行处理的病人数 |
| `--trajectories` | `standard_trajectory` | 逗号分隔的完整轨迹名；模型感知轨迹使用 `model_perception_trajectory` |
| `--methods` | `full_context_memory` | 逗号分隔的记忆方法 |
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
| `summary_memory` | `SummaryMemory` | 使用 Memo LLM 增量把每阶段融入紧凑结构化记忆 |
| `mem0_memory` | `Mem0Memory` | Mem0 抽取原子事实、更新向量库并检索 top-8 |
| `vector_memory` | `VectorMemory` | 不调用 Memo LLM，直接嵌入已释放阶段原文并检索 top-8 |
| `langmem_memory` | `LangMemMemory` | LangMem 提取、更新长期语义记忆并检索 top-8 |
| `graphiti_memory` | `GraphitiMemory` | Graphiti 增量写入时序知识图谱并混合检索 top-8；需要 Neo4j |

所有方法共享同一个 answer model；`summary_memory`、`mem0_memory`、`langmem_memory` 和 `graphiti_memory` 的记忆构建使用 `MEMO_OPENAI_*`，四种检索方法共享 `EMBEDDING_OPENAI_*`。每次运行按“病人 × 轨迹 × 方法”建立独立 namespace，并在轨迹开始前 `reset()`；阶段按顺序释放后才允许写入记忆。

```bash
bash scripts/run_step4.sh --all \
  --methods full_context_memory,summary_memory,vector_memory,mem0_memory,langmem_memory,graphiti_memory
```

运行 `graphiti_memory` 前需先启动 `.env` 指定的 Neo4j；其他方法不依赖 Neo4j。

#### 启动 Neo4j（仅 `graphiti_memory` 必需）

`graphiti_memory` 需要一个可通过 Bolt 访问的 Neo4j；其余 memo 方法不依赖 Neo4j。先在 `.env` 填写与实际数据库一致的配置：

```dotenv
GRAPHITI_NEO4J_URI=bolt://127.0.0.1:7687
GRAPHITI_NEO4J_USER=neo4j
GRAPHITI_NEO4J_PASSWORD=YourStrongPassword
```

##### Autodl 容器：当前部署方式

当前服务器将 Neo4j 原生安装在持久盘 `/root/autodl-tmp/neo4j`，数据、日志和配置均位于该目录。完整实验前执行：

```bash
cd /root/autodl-tmp/OralGPT/OralMemo
set -a; source .env; set +a
NEO4J_HOME=/root/autodl-tmp/neo4j

# 未运行时启动；已运行时该命令不会重复启动
$NEO4J_HOME/bin/neo4j start

# 检查状态、日志与 Bolt 连通性
$NEO4J_HOME/bin/neo4j status
tail -f $NEO4J_HOME/logs/neo4j.log
$NEO4J_HOME/bin/cypher-shell -a bolt://127.0.0.1:7687 \
  -u neo4j -p "$GRAPHITI_NEO4J_PASSWORD" "RETURN 1 AS ok;"
```

日常管理命令：

```bash
NEO4J_HOME=/root/autodl-tmp/neo4j
$NEO4J_HOME/bin/neo4j stop
$NEO4J_HOME/bin/neo4j restart
$NEO4J_HOME/bin/neo4j status
```

如果 Autodl 新容器中尚未部署 Neo4j，可在 `/root/autodl-tmp` 安装 Community `5.26.8`。**首次启动前**设置密码；已有数据目录不要重复执行设置初始密码：

```bash
NEO4J_VERSION=5.26.8
NEO4J_HOME=/root/autodl-tmp/neo4j
cd /root/autodl-tmp
curl -LO "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
tar -xzf "neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
mv "neo4j-community-${NEO4J_VERSION}" "$NEO4J_HOME"

# 仅空数据目录首次初始化时执行；密码至少 8 个字符。
$NEO4J_HOME/bin/neo4j-admin dbms set-initial-password YourStrongPassword

cat >> "$NEO4J_HOME/conf/neo4j.conf" <<'EOF'
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474
server.https.enabled=false
EOF

$NEO4J_HOME/bin/neo4j start
```

Autodl 当前配置会监听容器的网络接口；除非已额外配置访问控制，否则不要将 `7474` / `7687` 暴露到公网。

##### Ubuntu：Docker 部署

Ubuntu 上已有 Docker 时，以下是最简单的持久化部署方式。首次创建容器时设置密码；下面的 `YourStrongPassword` 必须替换：

```bash
docker run -d \
  --name oralmemo-neo4j \
  --restart unless-stopped \
  -p 127.0.0.1:7474:7474 \
  -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/YourStrongPassword \
  -v oralmemo_neo4j_data:/data \
  neo4j:5.26.8

# 完整实验前确认服务已运行
docker start oralmemo-neo4j
docker ps --filter name=oralmemo-neo4j
docker logs -f oralmemo-neo4j
```

连接检查：

```bash
docker exec oralmemo-neo4j cypher-shell \
  -u neo4j -p YourStrongPassword \
  "RETURN 1 AS ok;"
```

日常停止和启动不会删除图数据：

```bash
docker stop oralmemo-neo4j
docker start oralmemo-neo4j
```

`NEO4J_AUTH` 只在空数据卷首次初始化时生效。需要彻底重建数据库时，确认不再需要图数据后再执行：

```bash
docker rm -f oralmemo-neo4j
docker volume rm oralmemo_neo4j_data
```

Neo4j 启动完成后可访问 `http://127.0.0.1:7474`，使用用户名 `neo4j` 与配置密码登录。不要使用 `NEO4J_AUTH=none`，也不要提交真实密码。


### 评分指标（`report.py`）

| 指标 | 适用任务 | 说明 |
| --- | --- | --- |
| **ACC** | base 任务（感知/纵向证据/跨模态/记忆更新） | LLM 裁判二元判定准确率；另按任务类型、模态细分 |
| **ERS** | 全部任务（感知/纵向证据/跨模态/记忆更新/治疗） | benchmark 生成阶段预先筛选的 `selected_evidence` 中，被模型答案正确覆盖的证据数 / 证据总数；所有任务统一口径，另按任务类型、模态细分 |
| **TPS** | `treatment` | 所有诊断与治疗 evaluation 的 rubric 得分均值（百分比） |



### 产物（`outputs/.../evaluation/<轨迹>/<answer_model>/`）

- `<方法>/answers.json`：逐任务作答和实际 `memory_context`
- `<方法>/memory_metrics.json`：独立成本检查点；方法中途失败也保留已发生的调用统计
- `<方法>/report.json`：单方法评分及 `memory_metrics` 检查点
- `report.json` / `report.csv`：同一模型下的多方法结果和成本对比
- 汇总 `report.json` 同时记录 answer、verifier、memo 模型及地址和 `memory_methods`
- `memory_metrics` 包含写入/检索次数、总检索秒数、Memo LLM 输入/输出 token、embedding 调用/token、失败数和失败率；缓存命中的 Memo 请求不重复计费

> Mem0 的向量库持久化在方法独立的 `vector_store/`。LangMem 当前使用每条轨迹独立的进程内 Store。Graphiti 使用共享 Neo4j，但通过独立 `group_id` 隔离并在每条轨迹开始前只清理自己的 group。


## 五、Report 长程病例流水线（`report_pipeline/`）

在原有据之外，本流水线从 `reports/` 下的**文献长程病例报告 PDF** 自动构造与 `oralgpt_cmf_llamafactory_sft_dataset.json` **同构**的数据，但阶段轴改为**时间**（就诊/随访时间点，跨月至跨年），用于扩充数据来源并新增「跨时间点记忆 / 趋势追踪 / 治疗-结局」这一评测维度。

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
├── step1_report_trajectory/           # 报告时间点阶段化、标准轨迹与模型感知轨迹
│   └── run_perception_trajectory.py   # 仅重新感知图片 observation QA
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

# 用 .env 中的 ANSWER 模型生成模型感知轨迹；报告级并行数为 4
bash scripts/run_perception_trajectory_report.sh --all --num-workers 4

bash scripts/run_step4_report.sh --all --num-workers 1 \
  --methods full_context_memory \
  --answer-workers 2 --score-workers 1 --method-workers 1

# 先用一篇报告做测试
bash scripts/run_step0_step1_report.sh --limit 1 --num-workers 1

bash scripts/run_step2_step3_report.sh --limit 1 --num-workers 1 \
  --stage-workers 2 --task-workers 4

bash scripts/run_perception_trajectory_report.sh --limit 1 --num-workers 1

bash scripts/run_step4_report.sh --limit 1 --num-workers 1 \
  --methods full_context_memory \
  --answer-workers 1 --score-workers 1 --method-workers 1
```

Report 模型感知只重新回答带 `image_paths` 的 observation QA；无图 observation 和所有 evaluation QA 原样复制。单篇 Report 内按时间顺序串行处理，以便后续图片 QA 只能使用此前已经释放的文本 observation 和模型图片 observation；`--num-workers` 并行处理不同 Report。也可以覆盖 `.env` 中的 Answer 服务：

```bash
bash scripts/run_perception_trajectory_report.sh --limit 4 --num-workers 4 \
  --model gpt-5 --base-url https://api.openai.com/v1
```

输出路径与病人侧一致：

```text
outputs/report/<PDF stem>/trajectories/model_perception_trajectory/<answer_model>/
├── model_perception_trajectory.json
└── perception_report.json
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
- 模型感知轨迹：`--model`、`--base-url` 可覆盖 `.env` 的 `ANSWER_OPENAI_MODEL` 和 `ANSWER_OPENAI_BASE_URL`；同一 Report 内图片 QA 保持串行。
- Step4：`--methods`、`--answer-model`、`--answer-base-url`、`--answer-workers`、`--score-workers`、`--method-workers`。

Step0 摄取、时间线抽取和 Step1 轨迹分别检查已有产物并自动续跑。Step4 支持 method 级答案和评分续跑。三个 report 脚本会自动激活 `cmfbench`，并将额外命令行参数传给对应 Python 入口。

### 产物（`outputs/report/<PDF stem>/`）

| 路径 | 说明 |
| --- | --- |
| `raw/`、`images/` | Step0 的 PDF 解析结果与图片 |
| `timeline.extracted.json`、`verification_report.json` | 时间线抽取和校验记录 |
| `trajectories/standard_trajectory.json`、`dataset_entry.json` | Step1 标准轨迹和 SFT 条目 |
| `trajectories/model_perception_trajectory/<answer_model>/` | 模型感知轨迹及图片 QA 的 `perception_report.json` |
| `evidence/evidence.json`、`graph/` | Step2 证据及证据图 |
| `tasks/`、`rubrics/` | Step3 benchmark 任务和评分 rubric |
| `evaluation/standard_trajectory/<answer_model>/` | Step4 评估根目录；各方法的答案和评分位于 `<方法>/`，多方法汇总直接位于该目录 |
