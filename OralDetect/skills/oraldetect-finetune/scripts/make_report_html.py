#!/usr/bin/env python
"""Render the run's report as one self-contained HTML page.

    python make_report_html.py --work-dir <OUT> --train <json> --val <json> \
        --title "在 <dataset> 数据集上微调 OralDetect" \
        --subject-cn 菌斑 --finding "<one sentence>" --out <REPORT>/report.html

Everything the page shows is read back from the run's own artefacts -- the two COCO
files, `step6_metrics.txt`, the training log, the two prediction files and the two
PNGs. Nothing is passed in by hand except the title and the one-sentence finding,
because those are judgement, not measurement.

The figures are embedded as JPEG data URIs: the published page has no network access
for images, so a `<img src="file://...">` or a CDN URL renders as a broken box.
"""
import argparse
import base64
import glob
import io
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict

TOKEN = re.compile(r"\{\{([A-Z_0-9]+)\}\}")


# ---------------------------------------------------------------- dataset stats
def coco_stats(path):
    d = json.load(open(path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per_img_boxes = defaultdict(int)
    per_img_cats = defaultdict(set)
    for a in d["annotations"]:
        per_img_boxes[a["image_id"]] += 1
        per_img_cats[a["image_id"]].add(a["category_id"])
    n_img = len(d["images"])
    used = set().union(*per_img_cats.values()) if per_img_cats else set()
    return {
        "images": n_img,
        "boxes": len(d["annotations"]),
        "n_cats": len(used),
        "cat_names": [cats[c] for c in sorted(used)],
        "avg_boxes": len(d["annotations"]) / n_img if n_img else 0.0,
        # counted over images that carry annotations, so an unlabelled image does
        # not silently drag the average toward zero
        "avg_cats": (sum(len(v) for v in per_img_cats.values()) / len(per_img_cats)
                     if per_img_cats else 0.0),
    }


# ---------------------------------------------------------------- metrics table
def parse_metrics(path):
    """Read the table compare_runs.py wrote. Its format is ours, so it is parsable."""
    txt = open(path).read()
    out = {}
    pats = {
        "tp":   r"TP\s+@score [\d.]+\s+([\d.-]+)\s+([\d.-]+)",
        "fp":   r"FP\s+@score [\d.]+\s+([\d.-]+)\s+([\d.-]+)",
        "fn":   r"FN\s+@score [\d.]+\s+([\d.-]+)\s+([\d.-]+)",
        "prec": r"precision @bestF1\s+([\d.]+)\s+([\d.]+)",
        "rec":  r"recall @bestF1\s+([\d.]+)\s+([\d.]+)",
        "f1":   r"best F1\s+([\d.]+)\s+([\d.]+)",
        "thr":  r"\(at score threshold\)\s+([\d.]+)\s+([\d.]+)",
        "map":  r"mAP @\[\.50:\.95\]\s+([\d.]+)\s+([\d.]+)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        if not m:
            sys.exit(f"FATAL: could not read `{k}` out of {path} -- was it written by "
                     f"compare_runs.py?")
        out[k] = (float(m.group(1)), float(m.group(2)))
    return out


# ---------------------------------------------------------------- training curve
def parse_curve(path):
    txt = open(path, errors="ignore").read()
    val = re.findall(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?coco/bbox_mAP: ([\d.]+)\s+"
                     r"coco/bbox_mAP_50: ([\d.]+)\s+coco/bbox_mAP_75: ([\d.]+)", txt)
    if not val:
        sys.exit(f"FATAL: no `Epoch(val)` lines with coco/bbox_mAP in {path}")
    loss = {int(e): float(l) for e, l in
            re.findall(r"Epoch\(train\) \[(\d+)\]\[\d+/\d+\].*?loss: ([\d.]+)", txt)}
    return [{"e": int(e), "mAP": float(a), "m50": float(b), "m75": float(c),
             "loss": loss.get(int(e))} for e, a, b, c in val]


def n_params(log_text_path=None, txt=None):
    if txt is None:
        txt = open(log_text_path, errors="ignore").read()
    m = re.search(r"(\d+) model params", txt)
    return m.group(1) if m else None


# ---------------------------------------------------------------- baseline mAP
def baseline_ap(gt_path, pred_path):
    """mAP / mAP50 / mAP75 for the released checkpoint -- the chart's dashed floor.

    Recomputed rather than scraped out of a log, because the eval logs live in
    different places depending on how the run was launched.

    maxDets matches mmdet's CocoMetric (proposal_nums 100/300/1000), NOT COCOeval's
    1/10/100 default -- the curve these values sit under is parsed from the mmengine
    log, and on a dense image the two conventions differ by a couple of mAP points.
    A dashed floor scored differently from the line above it would be a lie.
    """
    from contextlib import redirect_stdout
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    preds = json.load(open(pred_path))
    if not preds:
        return 0.0, 0.0, 0.0
    with redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        dt = gt.loadRes(preds)
        e = COCOeval(gt, dt, "bbox")
        e.params.maxDets = [100, 300, 1000]
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0]), float(e.stats[1]), float(e.stats[2])


# ---------------------------------------------------------------- image embedding
def img_uri(path, maxw, quality=92):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    # 4:4:4 -- the box outlines and their labels are one pixel wide, and chroma
    # subsampling smears exactly those
    im.save(buf, "JPEG", quality=quality, optimize=True, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- formatting
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def delta(before, after, decimals=3, higher_is_better=True):
    # subtract the ROUNDED values, so the column is the arithmetic the reader can
    # redo from the two cells beside it
    d = round(after, decimals) - round(before, decimals)
    good = (d > 0) if higher_is_better else (d < 0)
    sign = "+" if d > 0 else ("−" if d < 0 else "")
    body = f"{abs(d):.{decimals}f}" if decimals else f"{abs(d):.0f}"
    cls = ' class="up"' if good and d else ""
    return f'<td{cls}>{sign}{body}</td>'


def metric_row(cn, en, hint, before, after, big=True, decimals=3,
               higher_is_better=True, extra_cls=""):
    cls = " ".join(x for x in (("big" if big else ""), extra_cls) if x)
    attr = ' class="%s"' % cls if cls else ""
    hint_html = f'\n                  <span class="hint">{esc(hint)}</span>' if hint else ""
    fmt = (lambda v: f"{v:.{decimals}f}") if decimals else (lambda v: f"{v:.0f}")
    return (f'            <tr{attr}>\n'
            f'              <td><span class="metric"><span class="nm">{esc(cn)} '
            f'<span class="en">{esc(en)}</span></span>{hint_html}</span></td>\n'
            f'              <td>{fmt(before)}</td><td>{fmt(after)}</td>'
            f'{delta(before, after, decimals, higher_is_better)}\n'
            f'            </tr>')


# ---------------------------------------------------------------- pipeline record
def pipeline_rows(work_dir, log_text, train, val, tr, va, report_dir, check_txt):
    """Six rows, each with the evidence for its own tick.

    These used to be hardcoded ✓. A page that certifies its own process without
    looking is worse than one that omits the section: it reads as proof and is not.
    Every row below is a file that exists or a line that is in the log; anything that
    cannot be checked says so rather than showing a tick.
    """
    init = re.search(r"init\s+:\s+(\S+)", log_text)
    tower = re.search(r"BertModel LOAD REPORT.*?from:\s*(\S+)", log_text, re.S)
    clean = "load_from   : CLEAN ✅" in log_text or "load_from : CLEAN ✅" in log_text
    params = n_params(log_text_path=None, txt=log_text)
    epochs = len(re.findall(r"Epoch\(val\) \[\d+\]\[\d+/\d+\]", log_text))
    resolved = glob.glob(osp.join(work_dir, "*.py"))
    after = osp.join(work_dir, "eval_after", "val", "preds.bbox.json")
    before = osp.join(work_dir, "eval_before", "val", "preds.bbox.json")
    check_ok = osp.isfile(check_txt) and "usable as-is" in open(check_txt, errors="ignore").read()

    def ok(cond, yes, no, unknown=None):
        if cond is None:
            return ("unk", unknown or "— 无记录")
        return ("", "✓ " + yes) if cond else ("bad", "✗ " + no)

    steps = [
        ("标注转换", "格式识别与切分，训练 / 验证两份 COCO 标注",
         ok(osp.isfile(train) and osp.isfile(val),
            f'{tr["images"]} / {va["images"]}', "标注文件缺失")),
        ("模型权重", "检测器与文本编码器就位",
         ok((osp.exists(init.group(1)) and (tower is None or osp.exists(tower.group(1))))
            if init else None, "已验证", "权重路径已失效")),
        ("数据校验", "图片路径可读、无孤立标注、无异常框",
         ok(check_ok if osp.isfile(check_txt) else None, "通过", "未通过",
            "— 无 step3_check.txt")),
        ("启动配置", "训练配置解析并落盘到 work_dir",
         ok(bool(resolved), "已解析", "work_dir 内无解析后的配置")),
        ("基线打分与训练", "权重与模型结构逐参数比对，基线先于训练打分",
         ok(clean and osp.isfile(before),
            (f"{params} 个参数全对上" if params else "完全匹配") + f" · {epochs} 轮",
            "未见 CLEAN 或基线预测")),
        ("微调后打分", "同一份验证集、同一套评分标准，前后各跑一次",
         ok(osp.isfile(after), "已完成", "无微调后预测")),
    ]
    out = []
    for i, (name, what, (cls, label)) in enumerate(steps, 1):
        cls = f" {cls}" if cls else ""
        out.append(f'        <div class="step">\n'
                   f'          <span class="idx">{i} / 6</span>\n'
                   f'          <span class="what"><b>{esc(name)}</b> · {esc(what)}</span>\n'
                   f'          <span class="ok{cls}">{esc(label)}</span>\n'
                   f'        </div>')
    all_ok = all(c == "" for _, _, (c, _) in steps)
    return "\n".join(out), all_ok


def main():
    ap = argparse.ArgumentParser(description="Render the finetune report as one HTML page.")
    ap.add_argument("--work-dir", required=True, help="the run's work_dir ($OUT)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True, help="where to write the .html")
    ap.add_argument("--template", default=osp.join(osp.dirname(osp.abspath(__file__)),
                                                   "..", "assets", "report_template.html"))
    ap.add_argument("--title", help="the h1; defaults to the work_dir's name")
    ap.add_argument("--page-title", help="browser-tab / gallery name (keep it stable "
                                         "across redeploys)")
    ap.add_argument("--subject", default="dental plaque", help="class shown in the masthead")
    ap.add_argument("--subject-cn", default="目标", help="the class in the reader's language, "
                                                         "used in the metric hints")
    ap.add_argument("--photo-noun", default="照片",
                    help='what the images are, e.g. "口内照片" / "全景片"')
    ap.add_argument("--finding", required=True,
                    help="one sentence on what the comparison figure shows; <b> allowed")
    ap.add_argument("--metrics", help="default <work-dir>/report/step6_metrics.txt")
    ap.add_argument("--log", help="training log; default: the largest .log in <work-dir>")
    ap.add_argument("--preview", help="default <work-dir>/report/step3_preview.png")
    ap.add_argument("--compare", help="default <work-dir>/report/step6_compare.png")
    ap.add_argument("--preds-before", help="default <work-dir>/eval_before/val/preds.bbox.json")
    ap.add_argument("--gpu", default="", help='e.g. "1×A100-40G"')
    ap.add_argument("--minutes", type=int, help="wall-clock training minutes")
    ap.add_argument("--init", default="oraldetect.pth", help="the checkpoint training started from")
    ap.add_argument("--date", help="run date; default: the log's mtime")
    ap.add_argument("--standalone", action="store_true",
                    help="wrap the output in <!doctype html><html><head>…<body> so it is a "
                         "complete document to open or email. Leave it OFF for the copy you "
                         "publish as an Artifact -- that path supplies its own wrapper and "
                         "forbids these tags.")
    ap.add_argument("--jpeg-quality", type=int, default=92,
                    help="92 keeps the box labels legible; drop it for a lighter file "
                         "where only the layout matters")
    ap.add_argument("--max-width-compare", type=int, default=1900)
    ap.add_argument("--max-width-preview", type=int, default=1700)
    a = ap.parse_args()

    W = a.work_dir
    rep = osp.join(W, "report")
    a.metrics = a.metrics or osp.join(rep, "step6_metrics.txt")
    a.preview = a.preview or osp.join(rep, "step3_preview.png")
    a.compare = a.compare or osp.join(rep, "step6_compare.png")
    a.preds_before = a.preds_before or osp.join(W, "eval_before", "val", "preds.bbox.json")
    if not a.log:
        logs = sorted(glob.glob(osp.join(W, "*.log")), key=osp.getsize, reverse=True)
        if not logs:
            sys.exit(f"FATAL: no .log under {W} -- pass --log")
        a.log = logs[0]

    missing = [p for p in (a.template, a.train, a.val, a.metrics, a.log,
                           a.preview, a.compare, a.preds_before) if not osp.isfile(p)]
    if missing:
        sys.exit("FATAL: missing input:\n  " + "\n  ".join(missing))

    log_text = open(a.log, errors="ignore").read()
    tr, va = coco_stats(a.train), coco_stats(a.val)
    tot_img, tot_box = tr["images"] + va["images"], tr["boxes"] + va["boxes"]
    all_cats = sorted(set(tr["cat_names"]) | set(va["cat_names"]))
    M = parse_metrics(a.metrics)
    curve = parse_curve(a.log)
    b_map, b_50, b_75 = baseline_ap(a.val, a.preds_before)
    best = max(curve, key=lambda p: p["mAP"])
    plateau = next((p["e"] for p in curve
                    if p["e"] >= 5 and abs(best["mAP"] - p["mAP"]) <= 0.01), best["e"])

    pipe_html, pipe_ok = pipeline_rows(W, log_text, a.train, a.val, tr, va, rep,
                                       osp.join(rep, "step3_check.txt"))

    ck = sorted(glob.glob(osp.join(W, "best_*.pth")))
    ck_name = osp.basename(ck[0]) if ck else "best_*.pth"
    ck_size = f"{osp.getsize(ck[0]) / 2**20:.0f} M" if ck else "—"

    date = a.date or __import__("datetime").date.fromtimestamp(
        osp.getmtime(a.log)).isoformat()
    run_meta = " · ".join(x for x in (f"run {date}", a.gpu,
                                      f"{a.minutes} min" if a.minutes else "") if x)

    hero = [f"<span>起点 <b>{esc(a.init)}</b></span>"]
    if a.minutes:
        hero.append(f"<span>训练 <b>{a.minutes} 分钟</b></span>")
    if a.gpu:
        hero.append(f"<span>硬件 <b>{esc(a.gpu)}</b></span>")

    def ds_row(label, s, use, cls=""):
        attr = ' class="%s"' % cls if cls else ""
        return (f'            <tr{attr}>'
                f'<td>{esc(label)}</td><td>{s["images"]}</td><td>{s["boxes"]}</td>'
                f'<td>{s["avg_boxes"]:.1f}</td><td>{s["n_cats"]}</td>'
                f'<td>{s["avg_cats"]:.2f}</td><td>{esc(use)}</td></tr>')

    total = {"images": tot_img, "boxes": tot_box, "n_cats": len(all_cats),
             "avg_boxes": tot_box / tot_img if tot_img else 0,
             "avg_cats": (tr["avg_cats"] * tr["images"] + va["avg_cats"] * va["images"])
                         / tot_img if tot_img else 0}

    sub_cn = a.subject_cn
    metric_rows = "\n".join([
        metric_row("综合准确度", "mAP", "最常用的总分，1.0 为满分", *M["map"]),
        metric_row("F1 分数", "best F1", "查准与查全的综合", *M["f1"]),
        metric_row("查准率", "precision", f"模型标出来的，有多少确实是{sub_cn}", *M["prec"]),
        metric_row("查全率", "recall", f"实际存在的{sub_cn}，找出来了多少", *M["rec"]),
        metric_row("正确检出", "TP", "", *M["tp"], big=False, decimals=0,
                   extra_cls="rule-above"),
        metric_row("误报", "FP", "", *M["fp"], big=False, decimals=0,
                   higher_is_better=False),
        metric_row("漏检", "FN", "", *M["fn"], big=False, decimals=0,
                   higher_is_better=False),
    ])

    tree = [f'      <pre><span class="cm"># 模型和全部图表</span>',
            f'{W}/',
            f'├── <b>{ck_name}</b>      <span class="cm">微调后的模型，{ck_size}</span>',
            f'├── report/                              <span class="cm">全部图和表</span>',
            f'├── eval_before/val/preds.bbox.json      <span class="cm">微调前的预测，可离线重新打分</span>',
            f'├── eval_after/val/preds.bbox.json       <span class="cm">微调后的预测</span>',
            f'└── {osp.basename(a.log)}                    <span class="cm">完整训练日志</span>',
            '',
            '<span class="cm"># 配置（仓库内）</span>',
            'OralDetect/launch_bash/own_finetune.yaml · own_eval.yaml · own_finetune.slurm',
            'OralDetect/datas/instances_{train,val}.json · class_{names,texts}.json</pre>',
            '',
            '      <pre><span class="cm"># 重跑一次微调</span>',
            'sbatch -M gpu OralDetect/launch_bash/own_finetune.slurm',
            '',
            '<span class="cm"># 换个模型重新打分：改 own_eval.yaml 的 checkpoint / out_dir 后</span>',
            'sbatch -M gpu OralDetect/launch_bash/own_eval_after.slurm</pre>']

    cat_phrase = (f"一个标注类别" if len(all_cats) == 1 else f"{len(all_cats)} 个标注类别")
    fill = {
        "PAGE_TITLE":  esc(a.page_title or a.title or osp.basename(W.rstrip("/"))),
        "TITLE":       esc(a.title or osp.basename(W.rstrip("/"))),
        "SUBJECT":     esc(a.subject),
        "RUN_META":    esc(run_meta),
        "HERO_META":   "\n".join("        " + x for x in hero),
        "PIPELINE_ROWS": pipe_html,
        "DATASET_H2":  f"{tot_img} 张{a.photo_noun}，{cat_phrase}",
        "DATASET_ROWS": "\n".join([
            ds_row("训练集 train", tr, "调整模型"),
            ds_row("验证集 val", va, "打分，不参与训练"),
            ds_row("合计", total, "—", cls="emph")]),
        "PREVIEW_ALT": esc(f"训练集样本网格，每张{a.photo_noun}上用绿色矩形标出 "
                           f"{a.subject} 区域，每个框旁都有文字标签。"),
        "RESULT_H2":   f'微调前后，同一份 {va["images"]} 张验证图',
        "METRIC_ROWS": metric_rows,
        "CHART_SUB":   f'{va["images"]} 张图 · {va["boxes"]} 个框 · 每轮训练后打一次分',
        "CHART_NOTE":  ("三条线是同一个总分在三档判定宽严下的取值：浅色最宽松（框大致对上就算），"
                        "深色最严格（框要贴得很准），虚线是微调前的水平。"
                        f"分数在第 {plateau} 轮之后基本不再变化，说明训练已经收敛 —— "
                        "继续训练不会更好，要再提升需要更多数据。"),
        "CHART_ARIA":  esc(f"模型在验证集上的得分随训练轮次上升的曲线，三条判定宽严档位"
                           f"全部从微调前的虚线水平持续上升，并在第 {plateau} 轮后趋于平稳"),
        "FINDING":     a.finding,          # may carry <b>; author-supplied, not escaped
        "COMPARE_ALT": esc("三列对比图：左列为人工标注框，中列为微调前的预测，"
                           "右列为微调后的预测。每个框旁都标有类别与置信度。"),
        "CURVE_DATA":  json.dumps(curve, separators=(",", ":")),
        "BASE_M50":    f"{b_50:.3f}",
        "BASE_MAP":    f"{b_map:.3f}",
        "BASE_M75":    f"{b_75:.3f}",
        "DELIVERABLES": "\n".join(tree),
        "IMG_PREVIEW": img_uri(a.preview, a.max_width_preview, a.jpeg_quality),
        "IMG_COMPARE": img_uri(a.compare, a.max_width_compare, a.jpeg_quality),
    }

    tpl = open(a.template).read()
    need = set(TOKEN.findall(tpl))
    if need - set(fill):
        sys.exit(f"FATAL: template wants tokens this script does not fill: "
                 f"{sorted(need - set(fill))}")
    html = TOKEN.sub(lambda m: fill[m.group(1)], tpl)

    if a.standalone:
        # split at the first rendered element: everything above it is head material
        i = html.index('<header class="masthead">')
        html = ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n"
                + html[:i].rstrip()
                + "\n<style>img{max-width:100%}[hidden]{display:none!important}</style>\n"
                + "</head>\n<body>\n" + html[i:].rstrip() + "\n</body>\n</html>\n")

    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)

    print(f"  dataset     : {tot_img} images · {tot_box} boxes · {len(all_cats)} class(es)")
    print(f"  baseline    : mAP {b_map:.3f}  mAP50 {b_50:.3f}  mAP75 {b_75:.3f}")
    print(f"  finetuned   : mAP {M['map'][1]:.3f}  best F1 {M['f1'][1]:.3f} "
          f"(best epoch {best['e']}, plateau ~{plateau})")
    mode = "standalone document" if a.standalone else "artifact body (no doctype/html/body)"
    print(f"  wrote {a.out}  ({osp.getsize(a.out) / 1e6:.1f} MB, figures inlined, {mode})")
    print("  open it in a BROWSER -- in a text editor the inlined figures are a wall of base64")
    if not pipe_ok:
        print("  ⚠️  the pipeline section has a step that did not verify -- read it before "
              "handing the page over; it is reporting the run honestly, not failing to render")
    if osp.getsize(a.out) > 15.5e6:
        print("  ⚠️  over ~15.5 MB -- an Artifact must render under 16 MB. Re-run with a "
              "smaller --max-width-compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
