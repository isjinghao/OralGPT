from __future__ import annotations

from collections import defaultdict


# 每个 stage 接受的模态标签
# 运行时按每个病人的实际问答轮动态映射。缺失模态 → 对应 stage 留空。
STAGE_DEFS = [
    {
        "stage_id": "S0_PROFILE",
        "order": 0,
        "stage_type": "profile_text",
        "modality": ["TEXT_QA"],
        "labels": ["profile"],
    },
    {
        "stage_id": "S1_FP",
        "order": 1,
        "stage_type": "facial_photos",
        "modality": ["FP"],
        "labels": ["FP"],
    },
    {
        "stage_id": "S2_DP",
        "order": 2,
        "stage_type": "dental_photos",
        "modality": ["DP"],
        "labels": ["DP"],
    },
    {
        "stage_id": "S3_XR_XLA",
        "order": 3,
        "stage_type": "cephalometric_and_panoramic_xray",
        "modality": ["XR", "XLData"],
        "labels": ["ceph", "panoramic"],
    },
    {
        "stage_id": "S4_CT",
        "order": 4,
        "stage_type": "three_dimensional_ct",
        "modality": ["CT"],
        "labels": ["CT"],
    },
    {
        "stage_id": "S5_TMJ",
        "order": 5,
        "stage_type": "temporomandibular_joint",
        "modality": ["TMJ"],
        "labels": ["TMJ", "ECT"],
    },
]


def classify_turn(human: str) -> str:
    """依据 human 提问文本匹配模态, 返回标签之一: profile / FP / DP / ceph / panoramic / CT / TMJ / ECT / diagnosis / treatment。
    输入: human - 该轮 human 提问原文(可含 <image> 占位符)。
    输出: str - 模态标签。
    """
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
    # 9. 诊断(置于 profile 之后, 避免 primary concern 误判)
    if "what is the diagnosis" in t or "diagnosis based on" in t:
        return "diagnosis"
    # 10. 其余问诊轮均视为治疗规划
    return "treatment"


def build_patient_stages(source_turns: list[dict]) -> dict:
    """按临床释放顺序切分阶段(语义驱动, 动态映射 source_turn_ids)。

    功能: 先用 classify_turn 给每个问答轮打模态标签, 再按 STAGE_DEFS 把对应标签的轮次组装为 6 个阶段(含图片与 QA); 缺失模态的阶段 source_turn_ids/image_paths/qa_pairs 均留空。
        诊断轮 → heldout_diagnosis, 其余未归入检查阶段的轮(治疗等) → heldout_treatment。
        ECT 轮归入 S5_TMJ。
    输入: source_turns - build_source_turns 产出的轮次列表。
    输出: dict - 含 patient 信息、stages、heldout_turns 的患者阶段对象。
    """
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
        source_turn_ids = []
        for src in collected:
            turns.append(
                {
                    "source_turn_id": src["source_turn_id"],
                    "human": src["human"],
                    "assistant": src["assistant"],
                    "image_paths": src["image_paths"],
                }
            )
            image_paths.extend(src["image_paths"])
            source_turn_ids.append(src["source_turn_id"])

        stages.append(
            {
                "stage_id": stage_def["stage_id"],
                "order": stage_def["order"],
                "stage_type": stage_def["stage_type"],
                "modality": stage_def["modality"],
                "source_turn_ids": source_turn_ids,  # 缺失模态时为 []
                "image_paths": image_paths,
                "qa_pairs": turns,
            }
        )

    heldout = []
    for role, label in (("heldout_diagnosis", "diagnosis"), ("heldout_treatment", "treatment")):
        for src in sorted(label_to_turns.get(label, []), key=lambda s: s["source_turn_id"]):
            heldout.append(
                {
                    "source_turn_id": src["source_turn_id"],
                    "human": src["human"],
                    "assistant": src["assistant"],
                    "image_paths": src["image_paths"],
                    "role": role,
                }
            )

    return {
        "patient_id": source_turns[0]["patient_id"],
        "patient_name": source_turns[0]["patient_name"],
        "group": source_turns[0]["group"],
        "stages": stages,
        "heldout_turns": heldout,
    }
