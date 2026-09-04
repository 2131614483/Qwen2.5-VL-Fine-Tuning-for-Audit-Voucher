# QWEN2.5-VL-审计凭证微调 — 独立验收包

用 **Qwen2.5-VL-7B + LoRA（QLoRA）** 微调视觉语言模型，使用数据为模拟数据，自动识别记账凭证、银行回单、差旅报销单等审计票据，并发现不合规异常。全程在 **RTX 5060 Ti（16GB）** 上跑通。

---

## 📦 目录结构

```
QWEN2.5-VL-审计凭证微调-验收/
├── README.md                          ← 本文件（验收入口）
├── Qwen2.5-VL-7B-Instruct-bnb-4bit/   ← 4-bit 基座模型（6.9GB，全量）
├── 代码/
│   ├── train_bills.py                 ← 独立微调训练脚本
│   ├── test_bills.py                  ← 评估：一致率 + 字段校验 + 异常检测
│   ├── check_rules.py                 ← 规则引擎（判定异常，纯函数库）
│   ├── gen_report.py                  ← 实验报告生成器
│   ├── lora_model_bills/              ← 已训练好的 LoRA 适配器（165MB，可直接评估）
│   ├── lora_model_bills_merged/       ← 合并版（注：仅配置，权重需在目标机合并）
│   ├── eval_metrics.json              ← 最近一次评估指标（test_bills 输出）
│   └── train_bills_run*.log           ← 历史训练日志（gen_report 输入）
├── data_bills/                        ← 审计凭证数据集 + 生成器
│   ├── train.jsonl / val.jsonl        # 867 + 153 条（含异常标注）
│   ├── images/                        # 1020 张合成凭证图片（已含）
│   ├── _preview/                      # 各类样式预览图
│   ├── gen/                           # 合成生成器（可重新生成）
│   └── README.md                      # 数据说明
├── 实验报告/                          # v1-v4 训练复盘 + 实验详情
├── 审计票据凭证识别方案.md             # 早期方案
├── 微调实战教程.md                     # 原理 + 操作教程（VLM/LoRA/SFT）
└── RTX5060Ti环境配置方案.md            # 16GB 环境搭建指南
```

---

## ✅ 快速验收（无需重训）

```bash
# 0. 环境（需与训练同款：unsloth + transformers 5.x + torch cu128）
conda activate rtx5060tixunlian        # 或你自建的等价环境

# 1. 评估已训练模型（加载 lora_model_bills → 验证集推理 → 规则判定）
cd 代码
python test_bills.py
```

期望结果（已实测通过）：
- **规则判定异常：准确率 100% / 精确率 100% / 召回率 100%**（TP9 FP0 FN0 TN9）
- 模型文本自判异常：准确率 83.3% / 精确率 100% / 召回率 66.7%
- 输出写入 `代码/eval_metrics.json`

---

## 🔧 重新训练

```bash
cd 代码
python train_bills.py      # 完整训练约 30-40 分钟，输出 代码/lora_model_bills/
```

训练关键点（脚本内已配置）：
- 4-bit QLoRA，仅训语言层，r=16 / alpha=16，可训练参数 ~0.48%
- 票据文字密集 → 分辨率上限提至 `max_pixels=1024*28*28`
- 结束后用 `gen_report.py --version vX --train-log train_bills_runX.log` 生成实验报告

> 若机器非 RTX 5060 Ti 或显存不同：降低 `per_device_train_batch_size` / `gradient_accumulation_steps` 组合（有效 batch 保持 8）。

---

## 📊 数据（data_bills）

- **规模**：1020 张合成凭证（train 867 / val 153），5 类（记账凭证、银行回单、差旅报销单及其手写变体）
- **异常**：约 30% 样本 label=1，caption 标注【异常】原因（大写≠数字、借贷不平衡、缺字号/签名、账号位数错、未来日期、涂改等）
- **退化**：每张图随机叠加变形/损坏（透视、折痕、污渍、褪色、撕边、模糊、反光），severity 0-2
- **标注即真相**：生成时字段值就是标注，程序保证记账凭证借贷平衡
- 若要重新生成：`cd data_bills/gen && python build_jsonl.py`

---

## 📈 已达成指标（v3/v4，规则判定）

| 指标 | 值 |
|---|---|
| 规则判定异常 准确率/精确率/召回率 | 100% / 100% / 100% |
| 完全一致率（文本） | 66.7% |
| 正常样本字段错误 | 0 |
| 最终 loss | ~0.175 |

---

## ⚠️ 已知说明

- `lora_model_bills_merged/` 当前仅含配置（tokenizer/config），**不含 model.safetensors**：unsloth 在 Windows 上合并 4-bit 权重写盘存在兼容问题。如需合并模型，请在 Linux / AutoDL 上重跑 `save_pretrained_merged(..., save_method="forced_merged_4bit")`，或用 `lora_model_bills/`（LoRA 适配器，加载即用，本包评估即走此路）。
- 基座模型来自 ModelScope `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`，已下载入本包，无需联网。
