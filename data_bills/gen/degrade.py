# -*- coding: utf-8 -*-
"""图像退化库 — 审计实战场景（变形/损坏/成像质量）。severity 0=正常 1=中度 2=重度"""
import random
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import common

def perspective_warp(img, g=0.00045, h=0.0002):
    if random.random() < 0.25:
        return img
    g = random.uniform(-1, 1) * g
    h = random.uniform(-1, 1) * h
    coeffs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, g, h]
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

def gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.8, 2.2)))

def motion_blur(img):
    w, h = img.size
    dx = random.choice([0, 4, 6, -5, 8, -8])
    dy = random.choice([0, 0, 3, -3, 0, -4])
    if dx == 0 and dy == 0:
        dx = 5
    out = Image.new("RGB", img.size, (255, 255, 255))
    for k in range(5):
        out = Image.blend(out, ImageChops.offset(img, dx * k // 2, dy * k // 2), 0.2)
    return out

def _comp(img, ov):
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
def add_crease(img):
    w, h = img.size
    y = random.randint(int(h * 0.3), int(h * 0.7))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for i in range(-6, 7):
        a = int(40 * (1 - abs(i) / 7))
        d.line([(0, y + i), (w, y + i)], fill=(90, 90, 90, a))
    ov = ov.filter(ImageFilter.GaussianBlur(2))
    return _comp(img, ov)

def add_stain(img):
    w, h = img.size
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for _ in range(random.randint(1, 3)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(40, 130)
        c = random.choice([(80, 60, 40, 55), (100, 90, 70, 40), (60, 70, 90, 45), (50, 50, 50, 50)])
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    ov = ov.filter(ImageFilter.GaussianBlur(8))
    return _comp(img, ov)

def add_faded_band(img):
    w, h = img.size
    y = random.randint(int(h * 0.2), int(h * 0.8))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for i in range(-40, 41):
        a = int(150 * (1 - abs(i) / 40))
        d.line([(0, y + i), (w, y + i)], fill=(255, 255, 255, a))
    ov = ov.filter(ImageFilter.GaussianBlur(4))
    return _comp(img, ov)
def add_glare(img):
    w, h = img.size
    x = random.randint(int(w * 0.2), int(w * 0.8))
    y = random.randint(int(h * 0.2), int(h * 0.8))
    rx = random.randint(int(w * 0.2), int(w * 0.4))
    ry = random.randint(int(h * 0.12), int(h * 0.25))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(255, 255, 255, 85))
    ov = ov.filter(ImageFilter.GaussianBlur(10))
    return _comp(img, ov)

def torn_edge(img):
    w, h = img.size
    mask = Image.new("L", img.size, 255)
    d = ImageDraw.Draw(mask)
    cx, cy = random.choice([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    size = int(min(w, h) * random.uniform(0.1, 0.22))
    sx = 1 if cx == 0 else -1
    sy = 1 if cy == 0 else -1
    pts = [(cx, cy)]
    for i in range(1, 7):
        t = size * i / 6
        j = random.randint(-size // 5, size // 5)
        pts.append((cx + sx * t + j, cy + sy * t))
    pts.append((cx + sx * size, cy + sy * size))
    for i in range(5, 0, -1):
        t = size * i / 6
        j = random.randint(-size // 5, size // 5)
        pts.append((cx + sx * t, cy + sy * t + j))
    d.polygon(pts, fill=0)
    bg = Image.new("RGB", img.size, (232, 232, 232))
    return Image.composite(img, bg, mask)
def degrade_v2(img, severity=0):
    img = common.rotate_slight(img)
    img = common.add_noise(img)
    img = common.add_vignette(img)
    if severity >= 1:
        ops = [perspective_warp, gaussian_blur, add_crease, add_faded_band]
        for op in random.sample(ops, random.randint(1, 2)):
            if random.random() < 0.75:
                img = op(img)
    if severity >= 2:
        ops2 = [add_stain, torn_edge, motion_blur, add_glare]
        for op in random.sample(ops2, random.randint(1, 2)):
            img = op(img)
    return img

def draw_ink_alteration(img, x0, y0, x1, y1):
    """涂改痕迹：高不透明度的粗墨笔划，覆盖文本区域。与纸张损坏（半透明/模糊）明显区分。"""
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for _ in range(random.randint(4, 7)):
        sx = random.randint(x0, x1)
        sy = random.randint(y0, y1)
        ex = sx + random.randint(30, 90)
        ey = sy + random.randint(-12, 12)
        d.line([(sx, sy), (ex, ey)], fill=(20, 20, 20, random.randint(170, 220)), width=random.randint(10, 18))
    for _ in range(2):
        bx = random.randint(x0, x1)
        by = random.randint(y0, y1)
        br = random.randint(14, 26)
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=(25, 25, 25, 180))
    ov = ov.filter(ImageFilter.GaussianBlur(1.2))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
