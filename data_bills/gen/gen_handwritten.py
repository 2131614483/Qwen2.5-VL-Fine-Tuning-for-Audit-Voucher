# -*- coding: utf-8 -*-
"""
手写变体生成器
==============
复用记账凭证/差旅报销单模板，正文用楷体渲染并加重墨迹噪声，模拟手填票据。
"""
import os
import gen_voucher
import gen_receipt


def generate(kind="voucher", seed=None, anomaly=None):
    """kind: 'voucher' 手写记账凭证 / 'expense' 手写差旅报销单"""
    if kind == "voucher":
        img, fields = gen_voucher.generate(seed=seed, handwritten=True, anomaly=anomaly)
    else:
        img, fields = gen_receipt.generate(kind="expense", seed=seed, handwritten=True, anomaly=anomaly)
    fields["handwritten"] = True
    return img, fields


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "_preview")
    os.makedirs(out, exist_ok=True)
    for i in range(3):
        img, f = generate("voucher", seed=400 + i)
        img.save(os.path.join(out, f"hand_voucher_{i}.png"))
        print(f"手写凭证[{i}] {f['kind']} {f['date']} 借:{f['debit_total']:.2f}")
    for i in range(3):
        img, f = generate("expense", seed=500 + i)
        img.save(os.path.join(out, f"hand_expense_{i}.png"))
        print(f"手写报销单[{i}] {f['department']} {f['total']:.2f}")
    print("预览已保存")
