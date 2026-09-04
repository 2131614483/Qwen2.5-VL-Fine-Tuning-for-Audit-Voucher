# -*- coding: utf-8 -*-
"""
批量生成 + JSONL 构建
====================
调用各生成器批量产出凭证图片，并把字段 dict 转成与现有 data/train.jsonl
完全一致的格式：

    {"image": "images/xxx.jpg", "caption": "...", "instruction": "...", "category": "...", "label": 0}

用法（在 data_bills/gen/ 下运行）：
    python build_jsonl.py                    # 用默认数量
    python build_jsonl.py --n 记账凭证=200   # 覆盖某类数量
"""
import argparse
import json
import os
import random
import sys

import gen_voucher
import gen_receipt
import gen_handwritten
import degrade

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 类别 -> (生成函数, 建议数量)
DEFAULT_COUNTS = {
    "记账凭证": 400,
    "记账凭证_手写": 120,
    "银行回单": 220,
    "差旅报销单": 160,
    "差旅报销单_手写": 120,
}

INSTRUCTION = {
    "记账凭证": "你是一名专业的审计员。请准确识别并描述这张记账凭证中的所有关键信息，包括凭证类型、凭证字号、日期、附件张数、摘要、会计科目、借方金额、贷方金额、相关方签名等。",
    "银行回单": "你是一名专业的审计员。请准确识别并描述这张银行回单中的所有关键信息，包括银行名称、交易日期时间、付款方、付款账号、收款方、收款账号、金额（大写和小写）、摘要、流水号等。",
    "差旅报销单": "你是一名专业的审计员。请准确识别并描述这张差旅报销单中的所有关键信息，包括报销部门、报销日期、出差区间、事由、费用明细、金额合计、相关方签名等。",
}

# 每个类别的异常（不合规）类型，用于 label=1 样本
ANOMALIES = {
    "记账凭证": ["upper_mismatch", "imbalance", "missing_serial", "future_date", "alteration"],
    "银行回单": ["upper_mismatch", "missing_flow", "bad_account", "alteration"],
    "差旅报销单": ["total_mismatch", "missing_signature", "late_trip", "alteration"],
}
ANOMALY_RATIO = 0.40  # 异常（不合规）样本占比


def gen_func(kind):
    if kind == "记账凭证":
        return lambda seed, anomaly=None: gen_voucher.generate(seed=seed, anomaly=anomaly)
    if kind == "记账凭证_手写":
        return lambda seed, anomaly=None: gen_handwritten.generate("voucher", seed=seed, anomaly=anomaly)
    if kind == "银行回单":
        return lambda seed, anomaly=None: gen_receipt.generate("receipt", seed=seed, anomaly=anomaly)
    if kind == "差旅报销单":
        return lambda seed, anomaly=None: gen_receipt.generate("expense", seed=seed, anomaly=anomaly)
    if kind == "差旅报销单_手写":
        return lambda seed, anomaly=None: gen_handwritten.generate("expense", seed=seed, anomaly=anomaly)
    raise ValueError(kind)


def base_category(kind):
    return kind.replace("_手写", "")


def build_caption(fields):
    """由字段 dict 构建审计描述文本"""
    kind = fields["kind"]
    if kind == "记账凭证" or kind in ("收款凭证", "付款凭证", "转账凭证"):
        rows_txt = "；".join(
            f"{'借' if r['debit'] is not None else '贷'}：{r['subject']} ￥{r['debit'] if r['debit'] is not None else r['credit']:,.2f}"
            for r in fields["rows"]
        )
        serial_txt = f"{fields['prefix']}字第{fields['serial']:04d}号" if fields["serial"] is not None else "凭证字号缺失"
        return (
            f"{kind}，凭证字号：{serial_txt}，"
            f"日期：{fields['date']}，附件{fields['attach']}张。"
            f"摘要：{fields['rows'][0]['summary']}。{rows_txt}。"
            f"借方合计￥{fields['debit_total']:,.2f}，贷方合计￥{fields['credit_total']:,.2f}。"
            f"制单：{fields['maker']}，审核：{fields['auditor']}，"
            f"记账：{fields['bookkeeper']}，出纳：{fields['cashier']}。"
        )
    if kind == "银行回单":
        return (
            f"银行回单，银行：{fields['bank']}，交易日期：{fields['txn_date']}，"
            f"交易时间：{fields['txn_time']}。付款方：{fields['payer']}，"
            f"付款账号：{fields['payer_acct']}；收款方：{fields['payee']}，"
            f"收款账号：{fields['payee_acct']}。金额（大写）：{fields['upper']}，"
            f"金额：¥{fields['amount']:,.2f}。摘要：{fields['summary']}。"
            f"流水号：{fields['flow_no'] or '缺失'}。"
        )
    if kind == "差旅报销单":
        rows_txt = "；".join(
            f"{r['date'][-5:]} {r['item']} ￥{r['amount']:,.2f}" for r in fields["rows"]
        )
        return (
            f"差旅报销单，报销部门：{fields['department']}，报销日期：{fields['date']}，"
            f"出差区间：{fields['start_date']} 至 {fields['end_date']}，"
            f"事由：{fields['reason']}。费用明细：{rows_txt}。"
            f"合计：￥{fields['total']:,.2f}（大写：{fields['upper']}）。"
            f"报销人：{fields['employee'] or '缺失'}，部门经理：{fields['manager']}，"
            f"财务审核：{fields['finance']}。"
        )
    raise ValueError(kind)


def save_sample(category, label, prefix, idx, anomaly=None):
    """生成图片 + 返回 JSONL 行 dict。按随机严重度叠加变形/损坏退化。"""
    seed = 100000 + idx
    gen = gen_func(category)
    img, flds = gen(seed, anomaly=anomaly)
    # 正常样本只允许轻度退化（0/1），避免重度损坏被误判为"涂改"；
    # 异常样本允许中重度（1/2）。
    if label == 1:
        severity = random.choices([1, 2], [0.5, 0.5])[0]
    else:
        severity = random.choices([0, 1], [0.5, 0.5])[0]
    img = degrade.degrade_v2(img, severity)
    fname = f"{prefix}_{idx:05d}.jpg"
    img.convert("RGB").save(os.path.join(IMG_DIR, fname), quality=88)
    cap = build_caption(flds)
    note = flds.get("anomaly_note")
    if note:
        cap += f"【异常】{note}。"
    else:
        cap += "各关键字段核对一致，未发现异常。"
    return {
        "image": f"images/{fname}",
        "caption": cap,
        "instruction": INSTRUCTION[base_category(category)],
        "category": base_category(category),
        "label": label,
    }


PREFIX = {
    "记账凭证": "acc",
    "记账凭证_手写": "acc_hand",
    "银行回单": "bank",
    "差旅报销单": "exp",
    "差旅报销单_手写": "exp_hand",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", action="append", default=[], help="覆盖数量，如 记账凭证=200")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例")
    args = parser.parse_args()

    counts = dict(DEFAULT_COUNTS)
    for kv in args.n:
        k, v = kv.split("=")
        counts[k] = int(v)

    train, val = [], []
    idx = 0
    for category, count in counts.items():
        prefix = PREFIX[category]
        cat_samples = []
        for i in range(count):
            label = 0
            anomaly = None
            if random.random() < ANOMALY_RATIO:
                label = 1
                anomaly = random.choice(ANOMALIES[base_category(category)])
            cat_samples.append(save_sample(category, label, prefix, idx, anomaly))
            idx += 1
        n_val = max(1, int(round(len(cat_samples) * args.val)))
        val.extend(cat_samples[:n_val])
        train.extend(cat_samples[n_val:])
        print(f"{category}: 共{count} 训{len(cat_samples)-n_val} 验{n_val}")

    with open(os.path.join(BASE_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(BASE_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n完成：训练 {len(train)} 条，验证 {len(val)} 条 → {BASE_DIR}/train.jsonl & val.jsonl")


if __name__ == "__main__":
    main()
