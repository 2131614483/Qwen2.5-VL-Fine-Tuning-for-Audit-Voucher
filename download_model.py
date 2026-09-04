"""
Qwen2.5-VL-7B-Instruct 模型下载脚本
使用 ModelScope 的 snapshot_download 函数下载模型

模型信息：
- 模型名称: unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
- 大小: 约 6.9GB (4-bit 量化, NF4)
- 用途: 视觉语言模型（QLoRA 基座），配合本项目在 16GB 显存上微调/推理

使用方法：
    python download_model.py

注意：
- 首次下载需要较长时间，请确保网络畅通
- 建议预留至少 15GB 磁盘空间
- 下载到项目根目录 models/ 下，训练/推理脚本默认读取该路径
"""

import os
from modelscope import snapshot_download

# 模型ID（4-bit 量化版：Qwen2.5-VL-7B-Instruct 的 NF4 量化，适合 16GB 显存 QLoRA）
MODEL_ID = 'unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit'

# 下载目录（模型将保存到 models/Qwen2.5-VL-7B-Instruct-bnb-4bit）
LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'Qwen2.5-VL-7B-Instruct-bnb-4bit')

# 确保目录存在
os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"=" * 60)
print(f"开始下载模型: {MODEL_ID}")
print(f"保存目录: {LOCAL_DIR}")
print(f"=" * 60)

try:
    # 使用 snapshot_download 下载模型
    # 参数说明：
    # - model_id: 模型在 ModelScope 上的 ID
    # - local_dir: 本地保存目录
    model_dir = snapshot_download(
        model_id=MODEL_ID,
        local_dir=LOCAL_DIR
    )
    
    print(f"\n" + "=" * 60)
    print(f"模型下载成功!")
    print(f"模型目录: {model_dir}")
    print(f"=" * 60)
    
    # 列出下载的文件
    print("\n下载的文件列表:")
    for root, dirs, files in os.walk(LOCAL_DIR):
        level = root.replace(LOCAL_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"{subindent}{file} ({file_size:.2f} MB)")
            
except Exception as e:
    print(f"\n下载失败: {e}")
    print("请检查网络连接或重试")
    raise