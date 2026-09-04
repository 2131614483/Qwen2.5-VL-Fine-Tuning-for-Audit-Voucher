# -*- coding: utf-8 -*-
"""
训练实验报告生成器
=================
每次训练跑完后，从训练日志 + 评估指标 + 数据统计，自动生成详细 markdown 实验报告。
用法（在 代码/ 目录下运行）：
    python gen_report.py --version v3 --train-log train_bills_run3.log --notes notes.md
输出：实验报告/训练实验-{version}-{日期}.md
"""
import argparse, json, os, re, datetime
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data_bills")
OUT_DIR = os.path.join(BASE, "实验报告")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data_stats():
    stats = {}
    for split in ("train", "val"):
        path = os.path.join(DATA_DIR, split + ".jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        cats = Counter(r["category"] for r in rows)
        labels = Counter(r["label"] for r in rows)
        stats[split] = {"n": len(rows), "cats": dict(cats), "labels": dict(labels)}
    return stats


def parse_train_log(path):
    txt = open(path, encoding="utf-8").read().replace("\r", "\n")
    loss = re.search(r"训练完成！损失: ([0-9.]+)", txt)
    steps = re.search(r"Total steps = (\d+)", txt)
    examples = re.search(r"Num examples = (\d+)", txt)
    gpu = re.search(r"GPU: (.+?),", txt)
    return {
        "loss": loss.group(1) if loss else "N/A",
        "steps": steps.group(1) if steps else "N/A",
        "examples": examples.group(1) if examples else "N/A",
        "gpu": gpu.group(1) if gpu else "N/A",
    }


def load_metrics(path="eval_metrics.json"):
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def render_table(rows):
    lines = ["| " + " | ".join(str(c) for c in rows[0]) + " |",
             "|" + "---|" * len(rows[0])]
    for r in rows[1:]:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def build(version, tlog, metrics, notes_text, stats):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    t = parse_train_log(tlog)
    tr, va = stats["train"], stats["val"]
    tr_lab = str(tr["labels"].get(0, 0)) + "/" + str(tr["labels"].get(1, 0))
    va_lab = str(va["labels"].get(0, 0)) + "/" + str(va["labels"].get(1, 0))
    L = []
    L.append("---")
    L.append("tags:")
    L.append("  - 训练实验")
    L.append("  - 微调")
    L.append("  - 审计凭证")
    L.append("  - claude")
    L.append("version: " + version)
    L.append("date: " + now)
    L.append("---")
    L.append("")
    L.append("# 训练实验报告 — 审计凭证识别 " + version)
    L.append("")
    L.append("> 生成时间：" + now + " ｜ 数据集：data_bills ｜ 最终损失：" + t["loss"])
    L.append("")
    L.append("## 1. 数据")
    L.append(render_table([
        ["split", "样本数", "类别分布", "label 0/1"],
        ["train", str(tr["n"]), json.dumps(tr["cats"], ensure_ascii=False), tr_lab],
        ["val", str(va["n"]), json.dumps(va["cats"], ensure_ascii=False), va_lab],
    ]))
    L.append("")
    L.append("## 2. 训练")
    L.append(render_table([
        ["项", "值"],
        ["GPU", t["gpu"]],
        ["训练样本", t["examples"]],
        ["总步数", t["steps"]],
        ["最终损失", t["loss"]],
        ["训练时长", fmt_duration(parse_duration(tlog))],
    ]))
    L.append("")
    L.append("## 3. 评估指标")
    if metrics:
        total = metrics["total"]
        exact = metrics["exact"]
        L.append("- **完全一致率**：" + str(exact) + "/" + str(total) + " = " + f"{exact / total:.1%}")
        fec = metrics.get("field_err_count", {})
        L.append("- **字段错误分布**：" + json.dumps(fec, ensure_ascii=False))
        a = metrics.get("anomaly", {})
        tp, fp, fn, tn = a.get("tp", 0), a.get("fp", 0), a.get("fn", 0), a.get("tn", 0)
        acc = (tp + tn) / total if total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        L.append(f"- **异常检测**：准确率{acc:.1%} 精确率{prec:.1%} 召回率{rec:.1%} (TP{tp} FP{fp} FN{fn} TN{tn})")
        ar = metrics.get("anomaly_rule", {})
        rtp, rfp, rfn, rtn = ar.get("tp", 0), ar.get("fp", 0), ar.get("fn", 0), ar.get("tn", 0)
        if any((rtp, rfp, rfn, rtn)):
            racc = (rtp + rtn) / total if total else 0
            rprec = rtp / (rtp + rfp) if (rtp + rfp) else 0
            rrec = rtp / (rtp + rfn) if (rtp + rfn) else 0
            L.append(f"- **规则判定异常**：准确率{racc:.1%} 精确率{rprec:.1%} 召回率{rrec:.1%} (TP{rtp} FP{rfp} FN{rfn} TN{rtn})")
        L.append("")
        L.append("异常检测-按类别（异常检出/该类异常总数）：")
        abc = metrics.get("anomaly_by_category", {})
        cats = sorted(set(k.split("_")[0] for k in abc))
        for c in cats:
            tt = abc.get(c + "_tp", 0)
            ff = abc.get(c + "_fn", 0)
            L.append("- " + c + "：" + str(tt) + "/" + str(tt + ff))
    else:
        L.append("（未提供 eval_metrics.json，无量化指标）")
    L.append("")
    if notes_text.strip():
        L.append("## 4. 分析与结论")
        L.append(notes_text.strip())
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--metrics", default="eval_metrics.json")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()
    notes_text = ""
    if args.notes and os.path.exists(args.notes):
        notes_text = open(args.notes, encoding="utf-8").read()
    md = build(args.version, args.train_log, load_metrics(args.metrics), notes_text, load_data_stats())
    fname = "训练实验-" + args.version + "-" + datetime.date.today().isoformat() + ".md"
    out = os.path.join(OUT_DIR, fname)
    open(out, "w", encoding="utf-8").write(md)
    print("报告已生成: " + out)


# 程序入口：见文件末尾


def parse_duration(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"训练耗时: (\d+)分(\d+)秒", txt)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    els = re.findall(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\s*<", txt)
    if els:
        n = [int(x) for x in els[-1].split(":")]
        return (n[0] * 60 + n[1]) * 60 + n[2] if len(n) == 3 else n[0] * 60 + n[1]
    return None

def fmt_duration(sec):
    if sec is None:
        return "未记录"
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}小时{m}分" if h else f"{m}分{s}秒"

if __name__ == "__main__":
    main()
