"""
Qwen2.5-VL-7B LoRA 审计凭证识别 — 独立微调
============================================
复用医学影像微调的技术路线（Unsloth FastVisionModel + 4-bit QLoRA + SFTTrainer），
针对 data_bills 数据集做独立微调，与医学模型互不影响。

关键差异（相对 train.py）：
1. DATA_DIR → ../data_bills
2. instruction 改用每条样本自带的 sample["instruction"]（按凭证类别定制提示词）
3. 票据文字密集，提高图片分辨率上限（min/max_pixels），失败则静默降级
4. 输出独立 lora_model_bills/，不覆盖医学 lora_model/
"""

import os
import time
import torch
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer

# ==================== 全局路径 ====================

# 以本文件位置为锚点定位项目根，避免依赖运行时 cwd
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE, "data_bills")              # 验收目录内 data_bills
MODEL_DIR = os.path.join(_BASE, "Qwen2.5-VL-7B-Instruct-bnb-4bit")  # 验收目录内基座模型
SAVE_DIR = os.path.join(_BASE, "代码", "lora_model_bills")
os.makedirs(SAVE_DIR, exist_ok=True)

# 票据文字密集：提升图片分辨率上限（memory 不够时降到 768*28*28）
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

T0 = time.time()  # 记录脚本总时长起点

# ==================== 1. 加载模型 ====================

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_DIR,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 提高图片分辨率上限（Qwen2.5-VL 的 processor 支持动态 min/max_pixels）
try:
    tokenizer.image_processor.min_pixels = MIN_PIXELS
    tokenizer.image_processor.max_pixels = MAX_PIXELS
    print(f"图片分辨率上限已设为 max_pixels={MAX_PIXELS}")
except Exception as e:
    print(f"提示：未设置 max_pixels（使用默认），原因: {e}")

# ==================== 2. 添加 LoRA 适配器 ====================

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_rslora=False,
    loftq_config=None,
)

# ==================== 3. 加载数据集 ====================

dataset = load_dataset("json", data_files=f"{DATA_DIR}/train.jsonl", split="train")
print(f"数据集大小: {len(dataset)}")

# ==================== 4. 转换为对话模板 ====================

def convert_to_conversation(sample):
    # 每条样本用自己 JSONL 里的 instruction（按类别定制）
    instruction = sample["instruction"]
    image_path = os.path.join(DATA_DIR, sample["image"])
    pil_image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": pil_image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["caption"]}],
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
print(f"格式转换完成，样本数: {len(converted_dataset)}")

# ==================== 5. 微调前推理（对比用） ====================

print("\n========== 微调前推理 ==========")

FastVisionModel.for_inference(model)

sample_0 = dataset[0]
test_image = Image.open(os.path.join(DATA_DIR, sample_0["image"])).convert("RGB")
print(f"类别: {sample_0['category']} | 参考描述: {sample_0['caption'][:80]}...")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": sample_0["instruction"]},
        ],
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("模型输出：")
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.2,   # 审计要准不要花
    min_p=0.05,
)

# ==================== 6. 训练 ====================

print("\n========== 开始训练 ==========")

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=40,
        lr_scheduler_type="cosine",
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}, 总显存: {max_memory} GB")
print(f"训练前已预留显存: {start_gpu_memory} GB")

t_train0 = time.time()
trainer_stats = trainer.train()
train_sec = int(time.time() - t_train0)
total_sec = int(time.time() - T0)
print(f"训练完成！损失: {trainer_stats.training_loss:.4f}")
print(f"训练耗时: {train_sec // 60}分{train_sec % 60}秒")
print(f"脚本总耗时: {total_sec // 60}分{total_sec % 60}秒")

# ==================== 7. 微调后推理 ====================

print("\n========== 微调后推理 ==========")

FastVisionModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": sample_0["instruction"]},
        ],
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("模型输出：")
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.2,
    min_p=0.05,
)

# ==================== 8. 保存模型 ====================

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"LoRA 适配器已保存到: {SAVE_DIR}/")



# 现在先不合并，因为合并过程不可逆
# try:
    # 4-bit 基座合并必须用 forced_merged_4bit，merged_16bit 不会真正保存
    # model.save_pretrained_merged(f"{SAVE_DIR}_merged", tokenizer, save_method="forced_merged_4bit")
    # print(f"合并模型已保存到: {SAVE_DIR}_merged/")
# except RuntimeError as e:
#     print(f"合并模型保存失败（不影响 LoRA 推理）: {e}")
