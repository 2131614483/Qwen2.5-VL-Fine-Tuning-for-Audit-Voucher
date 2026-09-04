# -*- coding: utf-8 -*-
"""
增值税发票公开数据下载 + 筛选 + 脱敏（占位）
==============================================
当前会话外网工具不可用，此脚本先提供筛选/脱敏的核心函数，数据源就绪后补充下载逻辑。

公开数据源候选（落地时验证）：
- 飞桨 AI Studio（PaddleOCR 社区"增值税发票数据集"，多带 XML/JSON 字段框标注）
- ModelScope 发票识别数据集
- GitHub 上散落的发票数据集

⚠️ 合规提醒：真实发票含个人信息（姓名/手机/身份证/银行账号），
   下载使用必须脱敏，且仅作模型训练研究用途。
"""
import os
import hashlib
from PIL import Image, ImageFilter, ImageEnhance


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dedup_by_hash(paths):
    """按文件哈希去重，返回保留的路径列表"""
    seen = set()
    kept = []
    for p in paths:
        h = file_sha256(p)
        if h not in seen:
            seen.add(h)
            kept.append(p)
    return kept


def check_resolution(path, min_size=(512, 384)):
    """清晰度粗筛：低于最低分辨率直接丢弃"""
    with Image.open(path) as im:
        w, h = im.size
    return w >= min_size[0] and h >= min_size[1]


def blur_score(path):
    """模糊度评分（方差法，需 numpy）。返回越小越模糊，低于阈值丢弃。"""
    try:
        import numpy as np
    except ImportError:
        return None
    with Image.open(path).convert("L") as im:
        im = im.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(im, dtype=np.float32)
    return float(arr.var())


def mask_regions(path, boxes, out_path, color=(20, 20, 20)):
    """
    敏感信息脱敏：把 boxes（[(x0,y0,x1,y1), ...]，由发票 XML/JSON 字段框给出）
    覆盖为纯色块后另存。没有字段框时可用大模型检测文本框再打码。
    """
    with Image.open(path).convert("RGB") as im:
        for (x0, y0, x1, y1) in boxes:
            im.paste(color, (int(x0), int(y0), int(x1), int(y1)))
        im.save(out_path)


def process_invoice_folder(src_dir, dst_dir, min_size=(512, 384), blur_threshold=80):
    """
    发票目录处理管线：去重 -> 分辨率/模糊筛选 -> 复制到 dst_dir（脱敏在拿到字段框后另跑）
    返回筛选后文件数。
    """
    os.makedirs(dst_dir, exist_ok=True)
    paths = [os.path.join(src_dir, n) for n in os.listdir(src_dir)
             if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    paths = dedup_by_hash(paths)
    kept = 0
    for p in paths:
        if not check_resolution(p, min_size):
            continue
        score = blur_score(p)
        if score is not None and score < blur_threshold:
            continue
        dst = os.path.join(dst_dir, os.path.basename(p))
        with open(p, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        kept += 1
    print(f"输入 {len(paths)}（去重后），筛选保留 {kept} 张 → {dst_dir}")
    return kept


if __name__ == "__main__":
    # TODO(数据源可用后)：替换为实际下载逻辑，例如：
    #   download_invoices(url_list, raw_dir="data_bills/invoices_raw")
    #   process_invoice_folder("data_bills/invoices_raw", "data_bills/invoices_clean")
    #   mask 后接入 build_jsonl.py 转 JSONL
    print("占位脚本：请补充发票下载数据源后使用 process_invoice_folder() 做筛选。")
