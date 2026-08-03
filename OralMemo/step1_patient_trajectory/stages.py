from __future__ import annotations

from collections import defaultdict


# 每个 stage 接受的模态标签
# 运行时按每个病人的实际问答轮动态映射。缺失模态 → 对应 stage 留空。
STAGE_DEFS = [
    {
        "stage_id": "S0_PROFILE",
        "order": 0,
        "stage_type": "perception",
        "modality": ["TEXT_QA"],
        "labels": ["profile"],
    },
    {
        "stage_id": "S1_FP",
        "order": 1,
        "stage_type": "perception",
        "modality": ["FP"],
        "labels": ["FP"],
    },
    {
        "stage_id": "S2_DP",
        "order": 2,
        "stage_type": "perception",
        "modality": ["DP"],
        "labels": ["DP"],
    },
    {
        "stage_id": "S3_XR_XLA",
        "order": 3,
        "stage_type": "perception",
        "modality": ["XR", "XLData"],
        "labels": ["ceph", "panoramic"],
    },
    {
        "stage_id": "S4_CT",
        "order": 4,
        "stage_type": "perception",
        "modality": ["CT"],
        "labels": ["CT"],
    },
    {
        "stage_id": "S5_TMJ",
        "order": 5,
        "stage_type": "perception",
        "modality": ["TMJ"],
        "labels": ["TMJ", "ECT"],
    },
]


def classify_turn(human: str) -> str:
    """依据 human 提问文本匹配模态；诊断与治疗问题统一返回 treatment。"""
    t = human.lower()
    # 1. ECT(纯文本, 无图片): 紧随 TMJ 的髁突稳定性问诊。
    if "single-photon emission" in t or "(ect)" in t:
        return "ECT"
    # 2. CT 必须先于 ceph: 3D CT 题干含 "prior sagittal cephalometric" 会误命中 ceph。
    if "3d ct" in t or "craniofacial reconstruction" in t or "dentofacial deformity based on the provided" in t:
        return "CT"
    # 3. 头颅侧位/正位片
    if "cephalometric radiograph" in t or ("cephalometric" in t and "radiograph" in t):
        return "ceph"
    # 4. 全景片
    if "panoramic" in t:
        return "panoramic"
    # 5. 颞下颌关节临床检查
    if "temporomandibular joint" in t or "(tmj)" in t:
        return "TMJ"
    # 6. 面像
    if "facial photograph" in t:
        return "FP"
    # 7. 口内像 / 咬合牙列
    if "intraoral" in t or "occlusal and dental" in t:
        return "DP"
    # 8. 基础问诊(年龄/主诉/既往史)
    if (
        "basic information" in t
        or "primary concern" in t
        or "past medical" in t
        or "medical or surgical history" in t
        or "chief complaint" in t
    ):
        return "profile"
    # 9. 诊断与其余决策问题统一视为治疗规划
    return "treatment"


def build_patient_stages(source_turns: list[dict]) -> dict:
    """按临床释放顺序切分阶段；诊断和治疗统一进入 `S6_TREATMENT`。"""
    # 按模态标签归集轮次(保持原始 source_turn_id 顺序)。
    label_to_turns: dict[str, list[dict]] = defaultdict(list)
    for src in source_turns:
        label = classify_turn(src["human"])
        label_to_turns[label].append(src)

    stages = []
    for stage_def in STAGE_DEFS:
        collected = []
        for label in stage_def["labels"]:
            collected.extend(label_to_turns.get(label, []))
        # 同一阶段内按真实轮序排列(如 TMJ 轮先于 ECT 轮)
        collected.sort(key=lambda s: s["source_turn_id"])

        turns = []
        image_paths = []
        for src in collected:
            turns.append(
                {
                    "source_turn_id": src["source_turn_id"],
                    "human": src["human"],
                    "assistant": src["assistant"],
                    "image_paths": src["image_paths"],
                    "role": "observation",
                }
            )
            image_paths.extend(src["image_paths"])

        stages.append(
            {
                "stage_id": stage_def["stage_id"],
                "order": stage_def["order"],
                "stage_type": stage_def["stage_type"],
                "modality": stage_def["modality"],
                "image_paths": image_paths,
                "qa_pairs": turns,
            }
        )

    evaluation_turns = [
        {
            "source_turn_id": src["source_turn_id"],
            "human": src["human"],
            "assistant": src["assistant"],
            "image_paths": src["image_paths"],
            "role": "evaluation",
            "ask_after_stage": "S5_TMJ",
            "release_after_stage": "S6_TREATMENT",
        }
        for src in label_to_turns.get("treatment", [])
    ]
    if evaluation_turns:
        stages.append(
            {
                "stage_id": "S6_TREATMENT",
                "order": len(stages),
                "stage_type": "treatment",
                "modality": ["TEXT_QA"],
                "image_paths": [],
                "qa_pairs": evaluation_turns,
            }
        )

    return {
        "patient_id": source_turns[0]["patient_id"],
        "patient_name": source_turns[0]["patient_name"],
        "group": source_turns[0]["group"],
        "stages": stages,
    }
