# OralGPT-X-Bench 评估框架设计

> 代码位置：`OralGPT/OralGPT-X/oralgpt_x_bench/`  
> **不修改** BAGEL 官方代码；运行时从 BAGEL 根目录调用本文件夹脚本。

## 首版结构（已定稿）

```
OralGPT-X-Bench
├── cbct          edit_restoration   低剂量→标准 CBCT
├── ortho         edit_simulation    正颌术前→术后
├── mri           edit_translation   T1↔T2 模态转换（与 cbct/ortho 同级）
└── t2i           caption→图
```

## 数据策略

- **Git 仓库**：仅含 `benchmark/cbct/metadata.examples.json` + 5 对 PNG（~224KB）
- **全量评估**：评估机器上 `FULL_EVAL=1` 或运行 `tools/export_cbct_benchmark.py`

## 三阶段流程

```
run_*.sh → infer/gen_edit_mp.py → metrics/pixel.py → summarize/summarize.py
```

## 部署

```bash
export BAGEL_ROOT=/path/to/Bagel
export model_path=/path/to/checkpoint
export output_path=/path/to/results
bash oralgpt_x_bench/scripts/run_cbct.sh
```

完整设计见仓库根目录 `OralGPT-X/OralGPT-X_Benchmark_Design.md`（如存在）。
