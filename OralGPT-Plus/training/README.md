# OralGPT-Plus RL Training Code (Mini-o3 based)

Reinforcement learning training code for **OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis** (CVPR 2026).

The code is built on top of [Mini-o3](https://github.com/Mini-o3/Mini-o3) (a VeRL based multi-turn agentic RL framework) and extends it with a new visual tool, `mirror_grounding`, used for panoramic X-ray analysis.

## Contents

The full training source tree is packaged in `mini-o3-rl.tar.gz`. Extract it with

```bash
tar xzf mini-o3-rl.tar.gz
cd mini-o3
```

Key entry points after extraction:

- `mirror_train_7b.sh` / `mirror_train_3b.sh` run the PPO training with the `tool_crop_mirror` / `crop_mirror` configuration.
- `val.sh`, `val_opg.sh` run validation.
- `verl/` holds the modified VeRL framework.

## Main modifications over upstream Mini-o3

- Added a `<mirror_grounding>` visual tool. Async and SPMD vLLM rollouts route these calls through `mirror_image`, with tool trigger bookkeeping kept consistent with the existing crop flow.
- Reward scoring (`verl/utils/reward_score/general_qa_tool.py`, `general_qa_tool_mc.py`) judges `<mirror_grounding>` steps the same way as `<grounding>` for format and accuracy rewards.
- Training analytics treat `crop_mirror` like the crop tool so regex checks, statistics, and reward reporting stay accurate.
- Separate wandb counters log `<grounding>` and `<mirror_grounding>` usage in `verl/trainer/ppo/ray_trainer.py`.

See `tool_modification_summary.md` inside the archive for the detailed change log.

## Judge API key

Reward computation can call an external judge model. The API key is read from the
`API_KEY` environment variable (no key is bundled in the source):

```bash
export API_KEY="your_judge_api_key"
```
