import string
from collections import defaultdict

import pandas as pd

from .text_mcq import CustomTextMCQDataset
from ..smp import load, dump


class OralQA_ZH(CustomTextMCQDataset):
    """
    OralQA-ZH: 口腔专业纯文本多选题基准（MCQ，中文）。

    - 列格式：index, question, A, B, C, D, E, answer, category, split
    - 继承 CustomTextMCQDataset 以复用通用 MCQ 文本评估逻辑
    """

    # 汇总 / LaTeX 打印时的科目顺序：牙体→牙周→外科→修复→正畸→黏膜→儿童→影像→预防→流行→病理
    CATEGORY_ORDER = [
        "牙体牙髓病学",
        "牙周病学",
        "口腔颌面外科学",
        "口腔修复学",
        "口腔正畸学",
        "口腔黏膜病学",
        "儿童口腔医学",
        "口腔颌面医学影像诊断学",
        "口腔预防医学",
        "口腔流行病学",
        "口腔组织病理学",
    ]

    # 与 CATEGORY_ORDER 对应科目的英文缩写（日志 / 论文表头）
    CATEGORY_ABBREV_EN = {
        "牙体牙髓病学": "Endo",
        "牙周病学": "Perio",
        "口腔颌面外科学": "OMFS",
        "口腔修复学": "Prosth",
        "口腔正畸学": "Ortho",
        "口腔黏膜病学": "OMD",
        "儿童口腔医学": "PedDent",
        "口腔颌面医学影像诊断学": "OMFR",
        "口腔预防医学": "PrevDent",
        "口腔流行病学": "OralEpi",
        "口腔组织病理学": "OMFP",
        "Overall": "Overall",
    }

    DATASET_URL = {
        "OralQA-ZH": (
            "https://huggingface.co/datasets/OralGPT/OralQA-ZH/"
            "resolve/main/OralQA-ZH.tsv"
        ),
    }
    DATASET_MD5 = {"OralQA-ZH": "9d861f09093797d5f23a57a6168fab2b"}

    def __init__(self, dataset: str = "OralQA-ZH", **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def load_data(self, dataset):
        """优先从 Hugging Face 自动下载 OralQA-ZH.tsv；否则使用本地 LMUDataRoot()/OralQA-ZH.tsv。"""
        if dataset in self.DATASET_URL:
            return self.prepare_tsv(
                self.DATASET_URL[dataset],
                self.DATASET_MD5.get(dataset),
            )
        return super().load_data(dataset)

    def build_prompt(self, line):
        """构造给 LLM 的 OralQA-ZH 专用提示词。"""
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        options_prompt = "选项如下（Options）:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"

        prompt = (
            "你是一名口腔医学专业考试考生，当前在完成口腔相关的选择题。\n"
            "请认真阅读题干和选项，根据口腔医学专业知识选择最合适的答案。\n"
            "要求：\n"
            "1）只输出你最终选择的选项序号（大写字母），例如 A、B、C、D、E 或 AC、BD 等；\n"
            "2）不要输出任何解析、推理过程或多余文字；\n"
            "3）如果你认为有多个选项正确，请按字母顺序连续输出（例如 ACD）。\n\n"
        )

        prompt += f"题目（Question）:\n{question}\n\n"
        prompt += options_prompt

        msgs = [dict(type="text", value=prompt)]
        return msgs

    @staticmethod
    def _normalize_option_string(raw) -> str:
        """
        将模型输出或标注答案规范为可比较的选项串。

        - 只保留 A–Z，忽略标点、中文、空格等（兼容 “答案：AC”“A, C”）
        - 多选按字母序排序后拼接，使 AB 与 BA、AC 与 CA 视为同一答案
        - 与 GT 只做整串相等判断：AB 对 AC 为错（部分重合不得分）
        """
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ""
        s = str(raw).strip()
        # For models that output CoT blocks (e.g., "<think> ... </think>"),
        # only parse the content after the last closing think tag.
        if "</think>" in s.lower():
            lower_s = s.lower()
            pos = lower_s.rfind("</think>")
            s = s[pos + len("</think>") :].strip()

        letters = [ch for ch in s.upper() if ch in string.ascii_uppercase]
        if not letters:
            return ""

        seen = set()
        uniq = []
        for ch in letters:
            if ch not in seen:
                seen.add(ch)
                uniq.append(ch)
        uniq.sort()
        return "".join(uniq)

    @classmethod
    def _normalize_prediction(cls, raw: str) -> str:
        """兼容旧名；与 `_normalize_option_string` 一致。"""
        return cls._normalize_option_string(raw)

    def evaluate(self, eval_file, **judge_kwargs):
        """
        OralQA-ZH 专用评估：
        - 对 prediction、answer 用同一套选项串规范化（多选按字母序）
        - 仅当规范化后两串完全相等计为正确（AB vs AC 为错，部分重合不计分）
        - 输出 Overall 和按 category 的统计结果（与其他基准风格一致）
        """
        data = load(eval_file)

        data["prediction"] = data["prediction"].apply(self._normalize_option_string)
        data["answer"] = data["answer"].apply(self._normalize_option_string)

        tot = defaultdict(int)
        correct = defaultdict(int)

        for _, row in data.iterrows():
            pred = row["prediction"]
            ans = row["answer"]
            category = row.get("category", "Overall") or "Overall"

            # 统一 category 名称
            cat = str(category).strip()
            if not cat:
                cat = "Overall"

            tot["Overall"] += 1
            tot[cat] += 1

            if pred == ans:
                correct["Overall"] += 1
                correct[cat] += 1

        # 汇总为 DataFrame（科目顺序固定为 CATEGORY_ORDER，其余未见过的 category 按字母排在中间科目之后、Overall 之前）
        categories = self.latex_ordered_categories(dict(tot))

        res = {"Category": [], "tot": [], "acc": []}
        for cat in categories:
            t = tot[cat]
            c = correct[cat]
            acc = (c / t * 100.0) if t > 0 else 0.0
            res["Category"].append(cat)
            res["tot"].append(t)
            res["acc"].append(acc)

        df_res = pd.DataFrame(res)

        # 保存结果文件，与其他基准命名风格保持一致
        suffix = eval_file.split(".")[-1]
        result_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(df_res, result_file)

        return df_res

    @classmethod
    def latex_ordered_categories(cls, cat_to_acc: dict) -> list:
        """
        与 LaTeX 行、终端「列顺序」一致的科目顺序（中文 category 名）。
        cat_to_acc: category -> acc（或其它占位），仅使用其键集。
        """
        raw_cats = set(cat_to_acc) - {"Overall"}
        ordered = [c for c in cls.CATEGORY_ORDER if c in raw_cats]
        unknown = sorted(raw_cats - set(ordered))
        return ordered + unknown + (["Overall"] if "Overall" in cat_to_acc else [])

    @classmethod
    def abbrev_for_log_line(cls, cat_to_acc: dict) -> str:
        """逗号分隔的英文缩写，顺序与 latex_ordered_categories 一致。"""
        cats = cls.latex_ordered_categories(cat_to_acc)
        parts = [cls.CATEGORY_ABBREV_EN.get(c, c) for c in cats]
        return ", ".join(parts)

    @classmethod
    def supported_datasets(cls):
        return ["OralQA-ZH"]
