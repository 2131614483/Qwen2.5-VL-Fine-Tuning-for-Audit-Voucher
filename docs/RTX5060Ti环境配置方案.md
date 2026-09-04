# RTX 5060 Ti 环境配置方案

> 针对本项目 **Qwen2.5-VL-7B LoRA 医学影像微调** 的完整环境搭建指南。

---

## 1. 硬件分析

| 项目 | RTX 4090（原项目） | RTX 5060 Ti（你的卡） | 结论 |
|------|-------------------|----------------------|------|
| 架构 | Ada Lovelace | **Blackwell** | 需更高 CUDA 版本 |
| 显存 | 24 GB | **16 GB**（推荐） / 8 GB | 16GB 可跑，8GB 困难 |
| 计算能力 | 8.9 | **12.x** | CUDA ≥ 12.8 |
| 原项目训练占用 | ≈7.85 GB | — | 16GB 够用，8GB 会 OOM |

> ⚠️ **关键结论**：必须入 16GB 版本。若只有 8GB，需要进一步阉割量化到 QLoRA（双重量化）或换更小模型。

---

## 2. 系统 & 驱动

### 2.1 NVIDIA 驱动

Blackwell 架构需要非常新的驱动，推荐 ≥ 570.x：

```bash
# 查看当前驱动版本
nvidia-smi

# Ubuntu 安装新驱动（推荐 570+）
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-575
sudo reboot
```

验证：

```bash
nvidia-smi
# 应显示 CUDA Version: 12.8 或更高、Driver Version: 57x.xx、GPU: RTX 5060 Ti
```

### 2.2 CUDA Toolkit 12.8+

```bash
# 下载并安装 CUDA 12.8 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.28.03_linux.run
sudo sh cuda_12.8.0_560.28.03_linux.run

# 写入环境变量
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

---

## 3. Python 环境

### 3.1 创建虚拟环境

```bash
# 使用 Python 3.11+，兼容性最好
conda create -n qwen-vl python=3.11 -y
conda activate qwen-vl

# 或使用 venv
python3.11 -m venv qwen-vl
source qwen-vl/bin/activate
```

### 3.2 PyTorch（CUDA 12.8 版本）

```bash
# PyTorch 最新 nightly / 正式版，明确指定 CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```python
import torch
print(torch.__version__)           # >= 2.7.0
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name())# NVIDIA GeForce RTX 5060 Ti
```

### 3.3 核心依赖

```bash
# Unsloth（12GB以下显存的核心救命库，注意版本是否支持 5060 Ti 新构架）
pip install unsloth

# Hugging Face 全家桶
pip install transformers>=4.51.0
pip install datasets
pip install accelerate

# TRL（SFTTrainer）
pip install trl

# 量化 & 高效注意力（Blackwell 需要验证兼容性）
pip install bitsandbytes
pip install xformers

# 数据处理
pip install pillow
pip install qwen-vl-utils
```

### 3.4 一键安装（完整 requirements）

```bash
cat > requirements_rtx5060ti.txt << 'EOF'
torch>=2.7.0
torchvision>=0.20.0
torchaudio>=2.7.0
unsloth
transformers>=4.51.0
datasets
accelerate
trl
bitsandbytes
xformers
pillow
qwen-vl-utils
EOF

pip install -r requirements_rtx5060ti.txt
```

---

## 4. 模型下载

**推荐**：用项目根目录的下载脚本（等价于下面命令）：

```bash
cd 项目根
python download_model.py     # 从 ModelScope 下载 unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit → models/
```

或手动 ModelScope 下载 4-bit 量化版（≈6.9GB）：

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit', local_dir='./models/Qwen2.5-VL-7B-Instruct-bnb-4bit')
"
```

> 训练/推理脚本默认从项目根 `Qwen2.5-VL-7B-Instruct-bnb-4bit/` 读基座模型；下载到 `models/` 后需把 `MODEL_DIR` 指向实际路径（或复制到根目录同名文件夹）。

---

## 5. 显存压榨优化 (5060 Ti 16GB 适应清单）

由于 5060 Ti 仅有 16GB 显存（相比 4090 的 24GB 少了 1/3），需要精细控制：

### 5.1 QLoRA 双重量化（强烈推荐）

```python
# 对比原方案，加上这些省显存
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # 再压 20-30%
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="./models/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    quantization_config=bnb_config,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
```

### 5.2 训练参数降级配置

| 参数 | 原值（4090） | 优化值（5060 Ti 16GB） | 原因 |
|------|-------------|----------------------|------|
| `per_device_train_batch_size` | 2 | **1** | 减少峰值显存 |
| `gradient_accumulation_steps` | 4 | **8** | 补偿 batch size，总 effective batch 不变 |
| `max_steps` | 30 | 30 | 不变 |
| `r` (LoRA rank) | 16 | **8** | 再省显存，效果略降 |
| `lora_alpha` | 16 | **8** | 保持 r==alpha |
| `use_gradient_checkpointing` | `"unsloth"` | `"unsloth"` | 保持 |
| `bf16` | True | True | 保持，比 fp16 训练更省 |
| `image_pixels` | 默认 (512) | **384** | 用小图分辨率省显存 |

对应代码：

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r = 8,                # 从16降到8
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,    # 从2降到1
        gradient_accumulation_steps = 8,    # 从4升到8, 保持effective batch=8
        max_steps = 30,
        learning_rate = 2e-4,
        warmup_steps = 5,
        lr_scheduler_type = "linear",
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        # 对于 5060 Ti, 关闭这些能省显存
        logging_steps = 10,
        save_steps = 1000,            # 不频繁保存
    )
)
```

### 5.3 Flash Attention 2（显存 + 速度双重收益）

```bash
pip install flash-attn --no-build-isolation
```

然后修改参数：

```python
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="./models/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    attn_implementation="flash_attention_2",  # 新增
)
```

### 5.4 图片分辨率控制

```python
# 如果原始图片是高清的（如 X 光片通常 2000+ 像素），可以截断分辨率上限
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./models/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    min_pixels=128 * 28 * 28,    # 100352 pixels (降低约一半)
    max_pixels=512 * 28 * 28,    # 401408 pixels
)
```

---

## 6. 已知坑点 & 排查

### 6.1 Blackwell 兼容性

Unsloth 对新架构硬件的支持可能滞后，首次运行时关注报错。如果遇到：

```
CUDA error: no kernel image is available for execution on the device
```

说明当前 PyTorch / bitsandbytes / xformers 未包含 Blackwell-sm_120 的内核。——**立刻将所有相关包升到最新版 / nightly。**

```bash
# 强制升级所有CUDA相关包到nightly
pip uninstall unsloth bitsandbytes xformers -y
pip install unsloth --pre --upgrade
pip install bitsandbytes --pre --upgrade
pip install xformers --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 6.2 显存溢出 (OOM) 紧急处理

出现 OOM 时逐级降级，每做一步后重试：

1. 改 `batch_size=1`，`accumulation_steps=8`
2. 改 LoRA `r=4`
3. 降低图片最大分辨率：`max_pixels=256*28*28`
4. 把 MLP 关掉（`finetune_mlp_modules=False`）
5. 使用 CPU offload（`device_map="auto"` + `offload_folder="./offload"`）

### 6.3 查看实时显存

```python
# 在关键步骤后加这行检查
import torch
print(f"已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"已预留: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

---

## 7. 快速验证脚本

创建 `check_env.py`，跑通说明环境没问题：

```python
"""RTX 5060 Ti 环境检查脚本"""
import torch

# 1. GPU 信息
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. 核心包
import transformers
import unsloth
import bitsandbytes
print(f"transformers: {transformers.__version__}")
print(f"unsloth: {unsloth.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")

# 3. 简单加载测试
from unsloth import FastVisionModel
print("✓ 所有包导入成功，环境就绪")
```

```bash
python check_env.py
```

预期输出：

```
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA Version: 12.8
PyTorch: 2.7.x
VRAM Total: 15.x GB
transformers: 4.5x.x
unsloth: 2025.x.x
bitsandbytes: 0.4x.x
✓ 所有包导入成功，环境就绪
```
