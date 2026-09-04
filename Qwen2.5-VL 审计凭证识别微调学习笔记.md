# Qwen2.5-VL 审计凭证识别微调学习笔记

> 适用场景：用你自己的 RTX 5060 Ti（16GB）微调视觉语言模型，让 AI 自动识别记账凭证、银行回单、差旅报销单等审计票据，并发现不合规异常。
> 本文从"模型怎么工作"讲到"怎么训练、怎么评估"，配合本项目 `data_bills/` 和 `代码/train_bills.py` 一起看。

---

# 第一部分：原理篇

## 1. 你在整个项目里做的到底是什么

一个视觉语言模型（VLM）本身"看过"很多图、读过很多书。它能看懂普通发票，但你发给它的凭证可能版式特殊、字段密集、还带涂改。你的目标不是让它"懂常识"，而是让它**在你这套凭证上的输出变得又快又准**——这就是微调。

整体链路：

```
准备数据 → 转成对话格式 → 加载 4-bit 基座模型 → 挂 LoRA → SFTTrainer 训练 → 存 LoRA → 推理评估
   (JSONL+图片)       (messages)        (Qwen2.5-VL)      (只训 0.48%)     (学习)     (lora_model_bills)
```

## 2. 视觉语言模型（VLM）是怎么"看图"的

Qwen2.5-VL 由三块组成：

```
┌──────────────┐   ┌───────────────────────┐   ┌──────────────────┐
│ 视觉编码器    │   │ 视觉-语言投影层         │   │ 大语言模型(LLM)   │
│ (ViT,冻结)   │──▶│ (Vision Tower→LLM)    │──▶│ (7B,可微调)       │
│ 图片→patch    │   │ 视觉token→LLM空间      │   │ 理解+生成文字      │
└──────────────┘   └───────────────────────┘   └──────────────────┘
```

关键点：**图片不是一整张喂进去，而是被切成小方块（patch）**。比如 28×28 像素为一个 patch，每张图被切成的 patch 数量取决于图片分辨率。每个 patch 变成一个"视觉 token"，和文字 token 一样进入语言模型。

所以图像分辨率决定两件事：
- **清晰度**：patch 太小看不清票据上的小字
- **计算量/显存**：patch 越多，视觉 token 越多，越吃显存

Qwen2.5-VL 用 `min_pixels / max_pixels` 控制图片被缩放到多大（见第 8 节）。

## 3. 为什么需要微调？什么时候才值得微调

基座模型对**常见标准票据**已经认得不错（预训练见过海量发票）。微调的价值体现在：

| 信号 | 说明 | 本项目是否命中 |
|------|------|:--:|
| 特定版式/冷门凭证 | 记账凭证、审计底稿这类格式，prompt 写上天也读不出字段结构 | ✅ |
| 需要固定结构化输出 | 强制按固定顺序输出字段，微调比 prompt 可靠 | ✅ |
| 高并发批处理 | 每天上千张，输出一致性重要 | ✅ |
| 数字类字段幻觉 | 金额、编号错一个就是审计事故，微调吃进正确样例压住幻觉 | ✅ |

**先测再定原则**：如果只是普通发票，先用基座模型 + 写好的提示词 + `temperature=0.2` 试跑 20 张，算字段准确率。≥90% 就不用微调。本项目针对的是基座模型"读不出"的凭证，所以值得微调。

## 4. LoRA 与 QLoRA 原理（重点）

### 4.1 全量微调为什么贵
一个 7B 模型有 80 亿参数。全量微调要更新所有参数，显存光放模型就 >28GB（fp16），还要存梯度、优化器状态，16GB 卡根本跑不动。

### 4.2 LoRA：只学增量，不学原参数
LoRA（Low-Rank Adaptation，低秩适配）的核心洞察：微调时模型参数的"有效变化量"通常在一个低维子空间里。所以它**冻结原权重 W，只训练两个小矩阵 A 和 B**：

```
W' = W + ΔW ,  其中 ΔW = B·A
        ↑          ↑
   冻结不动       B: d×r, A: r×k,  r 远小于 d 和 k（本项目 r=16）
```

形象理解：原权重是一本大词典（不让你改），你在旁边贴了张很小的便利贴 B·A 记录"针对你的凭证要做出什么调整"。推理时两本一起读。

- **r（rank）**：便利贴的大小。r 越大容量越大（效果上限高）但更吃显存；本项目用 r=16。
- **lora_alpha**：调整量缩放系数，通常 r==alpha。
- 训练参数量：本项目 **40,370,176 / 8,332,536,832 ≈ 0.48%**。就这么点参数，效果却接近全量微调。

### 4.3 QLoRA：4-bit 量化 + LoRA
把基座模型本身压缩到 4-bit（NF4 格式），省下绝大部分显存，只对 LoRA 那 0.48% 的参数保持高精度训练。这就是为什么 16GB 卡能训 7B 模型。

## 5. Unsloth 是干什么的

Unsloth 是一个让大模型微调**更快、更省显存**的库，它的 `FastVisionModel` 把上面的流程封装成几行调用：

- `from_pretrained`：加载 4-bit 模型（等价于帮你配好 QLoRA 量化）
- `get_peft_model`：挂上 LoRA 适配器，指定只训练哪些层
- `for_training` / `for_inference`：切换训练/推理模式
- 内部用 Triton 内核融合、动态量化等技巧，通常比原版快 2 倍以上、省 50% 显存

本项目里基座模型是 `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`（ModelScope 下载的 4-bit 版），`Qwen2.5-VL-7B-Instruct-bnb-4bit/` 目录就是它。

## 6. SFT 监督微调：模型到底在学什么

SFT（Supervised Fine-tuning，监督微调）的意思是：给模型看"问题→标准答案"，让它学会模仿答案。训练用的 `SFTTrainer` 来自 TRL 库。

### 6.1 一条样本长什么样（本项目 data_bills/train.jsonl 的一行）

```jsonl
{"image": "images/acc_00001.jpg", "caption": "付款凭证，凭证字号：付字第0164号，...", "instruction": "你是一名专业的审计员。请准确识别并描述这张记账凭证中的所有关键信息...", "category": "记账凭证", "label": 0}
```

| 字段 | 含义 | 训练时作用 |
|------|------|-----------|
| image | 图片相对路径 | 喂给视觉编码器 |
| instruction | 角色指令（按类别定制） | 作为用户消息 |
| caption | 审计描述（标准答案） | 作为 assistant 消息，是**要学的目标** |
| category | 凭证类别 | 分类型评估 |
| label | 0正常/1异常 | 训练不用，评估异常检测用 |

### 6.2 转成对话后长什么样

`train_bills.py` 里的 `convert_to_conversation` 会把一行 JSONL 转成：

```python
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image", "image": pil_image},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": caption}]},
]
```

训练时，模型把 user 部分和图片一起读进去，然后预测 assistant 部分，**损失只在 assistant 部分计算**——它学的是"看到这张凭证，照着描述输出"，而不是复读提示词。

## 7. 数据格式与图片路径约定

- JSONL 里 `image` 是**相对路径**，基准是数据集目录（`data_bills/`），所以写 `images/xxx.jpg`
- `train_bills.py` 里 `DATA_DIR = "../data_bills"`，加载时 `os.path.join(DATA_DIR, sample["image"])`
- 训练脚本必须**从 `代码/` 目录运行**，因为 `../data_bills` 是相对 `代码/` 的

## 8. 图像分辨率（max_pixels）的权衡

```python
MIN_PIXELS = 256 * 28 * 28   # 最小像素数（不足就放大）
MAX_PIXELS = 1024 * 28 * 28  # 最大像素数（超过就缩小）
```

- 28×28 是 patch 尺寸，所以这两个值 = patch 数 × 28²
- 医学影像文字少、图也小，用 512×28² 够；**票据文字密集，本项目提到 1024×28²** 保证小字可读
- 代价：分辨率越高 → 视觉 token 越多 → 显存越大。16GB 卡这是安全上限；OOM 就降到 768 或 512
- 训练和推理必须用**相同的 min/max_pixels**（`train_bills.py` 和 `test_bills.py` 都已设好）

## 9. 异常检测与 label 字段

数据里约 30% 样本 `label=1`，caption 末尾标注`【异常】…`（如大写≠数字、借贷不平衡、涂改痕迹）。训练时 label 本身不参与 loss——模型从 caption 里学会"读到什么 + 哪里不合规"。label 的作用在**评估**：`test_bills.py` 可对每个输出判断是否提到异常，与真实 label 对比算异常检测准确率。

---

# 第二部分：操作篇

## 0. 整体流程一览

```
环境配置 → 数据准备 → 训练 → 推理测试 → 评估 → 迭代
   (一次)   (每条路)  (30分钟)   (几分钟)   (人工)   (调参/补数据)
```

## 1. 环境准备（一次性）

### 1.1 硬件要求

| 项目 | 要求 | 说明 |
|------|------|------|
| GPU | RTX 5060 Ti 16GB（本机） | 16GB 可跑，8GB 困难 |
| 驱动 | ≥ 570.x | Blackwell 架构需要新驱动 |
| CUDA | ≥ 12.8 | 本机 cu128 已装好 |

### 1.2 安装依赖（本机已装好，重装机器时用）

```bash
conda create -n qwen-vl python=3.11 -y
conda activate qwen-vl
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install unsloth transformers datasets accelerate trl bitsandbytes pillow
```

验证环境：

```python
import torch
from unsloth import FastVisionModel
print(torch.__version__, torch.cuda.is_available())  # 2.x + True
print(torch.cuda.get_device_name())                  # NVIDIA GeForce RTX 5060 Ti
```

## 2. 数据准备（两条路）

### 路线 A：直接用现成的（推荐先这样跑通）

```
data_bills/
├── train.jsonl    # 867 条训练样本
├── val.jsonl      # 153 条验证样本
├── images/        # 1020 张凭证图片
└── gen/           # 合成生成器（想扩数据时用）
```

- 训练集包含：记账凭证（含手写变体）、银行回单、差旅报销单
- 约 30% 样本 `label=1`（不合规），图片带随机变形/损坏（透视、折痕、污渍、模糊等）

### 路线 B：自己重新生成数据

```bash
cd data_bills/gen
python build_jsonl.py                   # 用默认数量生成 1020 条
python build_jsonl.py --n 记账凭证=600   # 覆盖某类数量，扩量
python build_jsonl.py --n 记账凭证=0     # 去掉某类
```

生成器脚本分工：

| 脚本 | 作用 |
|------|------|
| `common.py` | 字体/人民币大写/随机人名公司 |
| `degrade.py` | 变形/损坏库（透视、折痕、污渍、撕边、模糊、反光） |
| `gen_voucher.py` | 记账凭证（借贷平衡由程序保证）+ 异常 |
| `gen_receipt.py` | 银行回单/差旅报销单 + 异常 |
| `gen_handwritten.py` | 手写变体 |
| `build_jsonl.py` | 批量生成 + 转 JSONL + label 异常采样 |

### 2.1 怎么检查数据质量

```bash
# 看类别和 label 分布
python -c "import json; from collections import Counter; \
rows=[json.loads(l) for l in open(\"data_bills/train.jsonl\",encoding=\"utf-8\")]; \
print(Counter(r[\"category\"] for r in rows)); print(Counter(r[\"label\"] for r in rows))"
```

再用 `data_bills/_preview/` 里的预览图，或直接打开 `images/` 里的图，肉眼核对"图 ↔ caption"是否一致。
## 3. 训练

### 3.1 训练脚本 `代码/train_bills.py` 逐段讲解

（1）**加载 4-bit 基座模型** —— 读入 Qwen2.5-VL-7B 的 4-bit 量化版，省显存：

```python
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_DIR,       # 本地 4-bit 模型路径
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",   # 省显存：训练时不存中间激活
)
```

（2）**设置图片分辨率上限** —— 票据文字密集，比医学场景开更高：

```python
tokenizer.image_processor.min_pixels = 256 * 28 * 28
tokenizer.image_processor.max_pixels = 1024 * 28 * 28
```

（3）**挂 LoRA** —— 视觉编码器冻结，只训语言层（0.48% 参数）：

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # 冻结视觉
    finetune_language_layers=True,  # 训语言层
    r=16, lora_alpha=16, lora_dropout=0,
)
```

（4）**加载数据 + 转对话** —— 每条样本用自己 JSONL 里的 instruction（按类别定制），图片相对 `DATA_DIR` 加载：

```python
def convert_to_conversation(sample):
    instruction = sample["instruction"]   # 每类凭证有自己的提示词
    pil_image = Image.open(os.path.join(DATA_DIR, sample["image"]))
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": pil_image},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["caption"]}]},
    ]}
```

（5）**微调前先推理一次** —— 用基座模型看看它对凭证有多差，作为训练效果的对照。

（6）**SFTTrainer 配置**（关键超参都在这）：

```python
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,   # 单卡批大小
        gradient_accumulation_steps=4,   # 累积4步，等效 batch = 2×4=8
        num_train_epochs=3,              # 完整过3轮
        learning_rate=2e-4,
        warmup_steps=40, lr_scheduler_type="cosine",
        optim="adamw_8bit", weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",              # 训练中不存档，跑完统一存
    ),
)
```

（7）**保存** —— 训练完存 LoRA 适配器（轻量，推理加载即可）：

```python
model.save_pretrained("lora_model_bills")
tokenizer.save_pretrained("lora_model_bills")
```

### 3.2 怎么运行

```bash
cd 代码
python train_bills.py
```

必须从 `代码/` 目录运行（`DATA_DIR = "../data_bills"` 是相对 `代码/` 的）。

### 3.3 训练时看什么

终端会滚动显示 `loss`。正常现象：

- 前几步 loss 偏高（~3-4），随后稳定下降，最后 ~1 以内
- 每 ~5.7s 一步，本数据集 327 步/epoch，3 epoch 约 30 分钟
- 显卡占用约 15.9GB（满），这是正常的

### 3.4 调参指南

| 现象 | 处理 |
|------|------|
| **OOM（显存溢出）** | 依次降：`max_pixels` 1024→768→512；batch 2→1；LoRA `r` 16→8 |
| **loss 不下降** | 学习率 2e-4→1e-4；检查数据是否损坏、caption 是否为空 |
| **效果差/学不会某类** | 该类数据不足，用 `--n 类别=数量` 扩量；或检查该类 instruction 是否合理 |
| **推理乱编（幻觉）** | 温度降到 0.1~0.3；给 caption 里加"未识别到XX"的负例 |
| **只记得训练集** | 降低 epoch（3→2），或加一点验证集早停 |
## 4. 推理测试

### 4.1 推理脚本 `代码/test_bills.py`

加载训练好的 LoRA 适配器 + 基座模型，对 `val.jsonl` 每类抽几张跑一遍，**逐条打印"参考描述 vs 模型输出"**：

```bash
cd 代码
python test_bills.py
```

关键点：

```python
model, tokenizer = FastVisionModel.from_pretrained("lora_model_bills", load_in_4bit=True)
FastVisionModel.for_inference(model)          # 切到推理模式

# 推理参数：审计要准不要花
model.generate(**inputs,
    max_new_tokens=256,       # 输出长度够描述整张凭证
    temperature=0.2,          # 低温度 → 少幻觉、稳定
    min_p=0.05,
)
```

### 4.2 为什么审计场景温度要低

`temperature` 控制随机性：

- 温度 1.0+：有创意，但会**编造金额**（审计致命）
- 温度 0.1~0.3：输出确定、稳定，宁可少说不可说错

## 5. 评估（怎么判断模型学好了）

### 5.1 定性评估（每次必做）

每类抽 10 张，人工判断模型输出是否**不缺、不错、不编造**：

| 检查项 | 说明 |
|--------|------|
| 关键信息是否齐全 | 凭证字号、日期、金额、科目/收付方都在吗 |
| 金额是否与图一致 | 一个数字都不能错 |
| 是否编造 | 图里没有的字段，模型不能自己编出来 |
| 异常是否识别 | `label=1` 的样本，模型有没有指出问题 |

### 5.2 定量评估

（1）**关键字段 F1**：写脚本从模型输出和 caption 里各抽出"金额/日期/编号"，算精确匹配率。

（2）**异常检测准确率**（本项目数据自带 label）：判断模型输出里是否提到异常（含"异常/不一致/缺失/涂改"等词），与真实 `label` 对比：

```python
# 伪代码：每张验证集图片
pred_abnormal = "异常" in model_output or "不一致" in model_output or "缺失" in model_output
# 与 sample["label"] 比较，统计准确率
```

### 5.3 迭代

评估发现问题 → 回补数据（改生成器或扩量）→ 重训。通常 2~3 轮迭代后模型会稳定。

## 6. 常见问题排查（FAQ）

| 问题 | 原因与解决 |
|------|-----------|
| `CUDA error: no kernel image...` | 驱动/CUDA 太旧，Blackwell 需 CUDA≥12.8，升驱动到 570+ |
| 训练 OOM | 按 3.4 降级清单处理 |
| 合并模型保存失败 | 不影响推理，用 LoRA 适配器加载即可 |
| 模型不跟着 instruction 走 | 确认 `train_bills.py` 用的是 `sample["instruction"]`（本项目已改） |
| 图很模糊识别差 | 提高 max_pixels；生成时检查退化强度是否过大（`degrade.py` severity） |
| 某类凭证全学不会 | 该类样本太少，`--n 类别=数量` 扩量后重训 |

## 7. 进阶方向

- **结构化输出**：让模型输出 JSON（`{"科目":..., "金额":...}`），便于对接审计系统
- **异常检测专项**：把 label 作为分类头，训练一个二分类输出
- **多页凭证**：合同、审计底稿常是多页的，需要切片或 PDF 转图
- **API 化**：FastAPI + Gradio，拖图片自动识别
- **真实数据补充**：用脱敏真实票据 + `gen/fetch_invoices.py` 的筛选/脱敏管线，补合成数据覆盖不到的复杂版式

| 训练异常变慢 / 被系统杀掉 | **先查显存被谁占用**：`nvidia-smi` 看 VRAM。本机 Ollama/其他 AI 工具会常驻占掉 10+GB，导致训练只剩几 GB、慢好几倍甚至被杀。训练前卸载：`curl -X POST localhost:11434/api/generate -d "{\"model\":\"<模型名>\",\"keep_alive\":0}"` |
| 训练过程 CPU/GPU 忙但不提速 | 前几步含 Triton 内核编译属正常，跑几十步后会提速；若一直慢，多半是显存被占或后台进程抢资源 |
