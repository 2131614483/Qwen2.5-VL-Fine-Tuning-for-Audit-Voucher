"""
加载训练好的审计凭证 LoRA 模型，对 data_bills 验证集做推理 + 审计指标评估。

评估维度：
1. 完全一致率：输出是否与参考 caption 逐字一致
2. 字段级校验：日期/金额/凭证字号/流水号/借贷平衡（正则提取对比，抓数字幻觉）
3. 异常检测：输出是否提到异常关键词，与真实 label 对比（分类准确率/精确率/召回率）

用法：cd "core code" && python test_bills.py
"""
import os
import re
from collections import Counter
from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel
from transformers import TextStreamer
import check_rules

# 以脚本位置为锚点定位项目根，避免依赖运行时 cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_BASE, "data_bills")
LORA_DIR = os.path.join(_THIS_DIR, "lora_model_bills")
TEST_PER_CATEGORY = 6   # 每类 6 张（3 正常 + 3 异常）

ABNORMAL_KW = ["【异常】", "不一致", "缺失", "涂改", "不平衡", "不匹配"]


def predict_abnormal(text):
    return any(k in text for k in ABNORMAL_KW)


def dates(text):
    return set(re.findall(r"\d{4}-\d{2}-\d{2}", text))


def amounts(text):
    return {float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.\d{2}", text)}


def serials(text):
    return set(re.findall(r"[记收付转]字第\d+号", text)) | set(re.findall(r"\d{18}", text))


# ==================== 1. 加载 LoRA 模型 ====================

model, tokenizer = FastVisionModel.from_pretrained(LORA_DIR, load_in_4bit=True)
FastVisionModel.for_inference(model)
try:
    tokenizer.image_processor.min_pixels = 256 * 28 * 28
    tokenizer.image_processor.max_pixels = 1024 * 28 * 28
except Exception:
    pass

# ==================== 2. 采样（每类均衡 label） ====================

dataset = load_dataset("json", data_files=f"{DATA_DIR}/val.jsonl", split="train")
seen = {}
samples = []
for s in dataset:
    key = (s["category"], s["label"])
    if seen.get(key, 0) >= TEST_PER_CATEGORY // 2:
        continue
    seen[key] = seen.get(key, 0) + 1
    samples.append(s)
print(f"测试 {len(samples)} 个样本，分布: {Counter((s['category'], s['label']) for s in samples)}")

# ==================== 3. 推理 + 评估 ====================

metrics = {"total": 0, "exact": 0, "field_err": [], "anom": Counter(), "anom_cat": Counter(),
           "anom_rule": Counter(), "anom_rule_cat": Counter()}

for sample in samples:
    ref = sample["caption"].strip()
    test_image = Image.open(os.path.join(DATA_DIR, sample["image"])).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": sample["instruction"]},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(test_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    in_len = inputs["input_ids"].shape[1]
    out_tokens = model.generate(
        **inputs, streamer=text_streamer,
        max_new_tokens=256, use_cache=True, temperature=0.2, min_p=0.05,
    )
    output = tokenizer.decode(out_tokens[0][in_len:], skip_special_tokens=True).strip()

    # --- 1 完全一致 ---
    metrics["total"] += 1
    exact = (output == ref)
    if exact:
        metrics["exact"] += 1

    # --- 2 字段级校验 ---
    errs = []
    d_ref, d_out = dates(ref), dates(output)
    if d_ref and not d_ref <= d_out:
        errs.append(f"日期: 参考{d_ref} 输出{d_out}")
    a_ref, a_out = amounts(ref), amounts(output)
    if a_ref and not a_ref <= a_out:
        errs.append(f"金额: 参考{a_ref} 输出{a_out}")
    s_ref, s_out = serials(ref), serials(output)
    if s_ref and not s_ref <= s_out:
        errs.append("字号/流水号缺失或错误")
    if sample["category"] == "记账凭证" and sample["label"] == 0:
        tot = re.findall(r"合计￥([\d,]+\.\d{2})", output)
        if len(tot) >= 2:
            a, b = float(tot[0].replace(",", "")), float(tot[1].replace(",", ""))
            if abs(a - b) > 0.01:
                errs.append(f"借贷不平衡: {a} vs {b}")

    if not exact or errs:
        note = f"字段错误: {errs}" if errs else "仅文本格式差异（无字段错误）"
        metrics["field_err"].append((sample["category"], sample["image"], output, note))

    # --- 3 异常检测（模型自判） ---
    pred = predict_abnormal(output)
    true = bool(sample["label"])
    k = "tp" if (pred and true) else ("fp" if (pred and not true) else ("fn" if (not pred and true) else "tn"))
    metrics["anom"][k] += 1
    metrics["anom_cat"][(sample["category"], k)] += 1

    # --- 3b 异常检测（规则引擎，解耦判定） ---
    pred_rule, rule_issues = check_rules.judge(sample["category"], output)
    k_rule = "tp" if (pred_rule and true) else ("fp" if (pred_rule and not true) else ("fn" if (not pred_rule and true) else "tn"))
    metrics["anom_rule"][k_rule] += 1
    metrics["anom_rule_cat"][(sample["category"], k_rule)] += 1

    print(f"[{sample['category']}|{'异常' if true else '正常'}] 一致={'是' if exact else '否'} "
          f"字段错={errs or '无'} 模型判异常={pred} 规则判异常={pred_rule}{'('+','.join(rule_issues)+')' if rule_issues else ''}")

# ==================== 4. 汇总 ====================

print("\n========== 评估汇总 ==========")
n = metrics["total"]
print(f"完全一致率: {metrics['exact']}/{n} = {metrics['exact'] / n:.1%}")
print("字段错误分布:", dict(Counter(e[0] for e in metrics["field_err"])))
an = metrics["anom"]
tp, fp, fn, tn = an["tp"], an["fp"], an["fn"], an["tn"]
acc = (tp + tn) / n if n else 0
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
print(f"异常检测: 准确率{acc:.1%} 精确率{prec:.1%} 召回率{rec:.1%} (TP{tp} FP{fp} FN{fn} TN{tn})")
print("异常检测-按类别（异常检出/该类异常总数）:")
for cat in sorted(set(k[0] for k in metrics["anom_cat"])):
    c = metrics["anom_cat"]
    tot_a = c[(cat, "tp")] + c[(cat, "fn")]
    print(f"  {cat}: {c[(cat, 'tp')]}/{tot_a}")
ar = metrics["anom_rule"]
rtp, rfp, rfn, rtn = ar["tp"], ar["fp"], ar["fn"], ar["tn"]
racc = (rtp + rtn) / n if n else 0
rprec = rtp / (rtp + rfp) if (rtp + rfp) else 0
rrec = rtp / (rtp + rfn) if (rtp + rfn) else 0
print(f"规则判定异常: 准确率{racc:.1%} 精确率{rprec:.1%} 召回率{rrec:.1%} (TP{rtp} FP{rfp} FN{rfn} TN{rtn})")

print("\n========== 需人工核对（最多 10 个） ==========")
for cat, img, out, note in metrics["field_err"][:10]:
    print(f"\n[{cat}] {img}\n  问题: {note}\n  输出: {out[:160]}")

# ==================== 5. 导出指标 JSON（供实验报告用） ====================
import json
out_json = {
    "total": metrics["total"],
    "exact": metrics["exact"],
    "field_err_count": {k: v for k, v in Counter(e[0] for e in metrics["field_err"]).items()},
    "anomaly": dict(metrics["anom"]),
    "anomaly_by_category": {f"{c}_{k}": v for (c, k), v in metrics["anom_cat"].items()},
    "anomaly_rule": dict(metrics["anom_rule"]),
    "anomaly_rule_by_category": {f"{c}_{k}": v for (c, k), v in metrics["anom_rule_cat"].items()},
}
with open(os.path.join(_THIS_DIR, "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)
print("\n指标已保存到 eval_metrics.json")
