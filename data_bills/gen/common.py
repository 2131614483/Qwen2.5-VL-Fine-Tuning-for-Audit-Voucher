# -*- coding: utf-8 -*-
"""
data_bills 合成数据生成器 — 公共模块
====================================
提供字体查找、人民币大写转换、随机人名/公司/日期、图像退化（噪声/旋转/纸张质感）等工具。
Windows 本机生成，训练时随数据上传 AutoDL。
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# -------------------- 字体查找 --------------------

_FONT_CANDIDATES = {
    "song": [
        r"C:\Windows\Fonts\simsun.ttc",            # 宋体
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ],
    "hei": [
        r"C:\Windows\Fonts\simhei.ttf",            # 黑体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ],
    "kai": [
        r"C:\Windows\Fonts\simkai.ttf",            # 楷体（手写替身）
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ],
    "fang": [
        r"C:\Windows\Fonts\simfang.ttf",           # 仿宋
    ],
    "msyh": [
        r"C:\Windows\Fonts\msyh.ttc",              # 微软雅黑
    ],
}

_font_cache = {}


def find_font(kind="song"):
    """返回可用字体路径（按优先级），找不到则返回候选首位（运行时再报错）。"""
    for p in _FONT_CANDIDATES.get(kind, []):
        if os.path.exists(p):
            return p
    return _FONT_CANDIDATES.get(kind, [None])[0]


def font(kind, size):
    """带缓存的字体加载。kind: song/hei/kai/fang/msyh"""
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(find_font(kind), size)
    return _font_cache[key]


# -------------------- 人民币大写 --------------------

_CN_DIGITS = "零壹贰叁肆伍陆柒捌玖"
_CN_UNIT = ["", "拾", "佰", "仟"]
_CN_BIG = ["", "万", "亿", "兆"]


def _group_str(g):
    """4 位数字组 → 中文读数，如 '0500'→'零伍佰'，'0000'→''"""
    if g == "0000":
        return ""
    s = ""
    zero_pending = False
    for i, ch in enumerate(g):
        d = int(ch)
        pos = len(g) - 1 - i
        if d == 0:
            zero_pending = True
        else:
            if zero_pending:
                s += "零"
                zero_pending = False
            s += _CN_DIGITS[d] + _CN_UNIT[pos]
    return s


def rmb_upper(value):
    """金额 → 人民币大写。128000.00 → 壹拾贰万捌仟元整"""
    value = round(float(value), 2)
    int_part = int(value)
    frac = int(round((value - int_part) * 100))
    jiao, fen = divmod(frac, 10)

    if int_part == 0:
        int_str = "零"
    else:
        s = str(int_part)
        groups = []
        while s:
            groups.append(s[-4:])
            s = s[:-4]
        parts = []
        for gi, g in enumerate(groups):
            sub = _group_str(g)
            if sub:
                parts.append(sub + (_CN_BIG[gi] if gi < len(_CN_BIG) else ""))
            else:
                parts.append("")
        joined = "".join(reversed(parts))
        while "零零" in joined:
            joined = joined.replace("零零", "零")
        if joined.endswith("零"):
            joined = joined[:-1]
        if joined == "":
            joined = "零"
        int_str = joined

    result = int_str + "元"
    if jiao == 0 and fen == 0:
        result += "整"
    elif jiao == 0:
        result += "零" + _CN_DIGITS[fen] + "分"
    else:
        result += _CN_DIGITS[jiao] + "角"
        if fen:
            result += _CN_DIGITS[fen] + "分"
    return result


def money_lower(value):
    """金额 → 千分位小写，如 12800 → 12,800.00"""
    return f"{float(value):,.2f}"


# -------------------- 随机数据 --------------------

_SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄麹家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍舄璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
_GIVEN = "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀兰霞平刚桂英华玉忠文辉国庆燕红鹏斌海波山峰磊刚建华雪荣婷淑兰桂芝永强志明小霞金桂芳秀英丽华春梅凤英冬梅桂兰玉兰秀兰秀荣桂芳秀英丽华慧颖伟文龙静涛云燕红"

_COMPANIES = [
    "深圳市华创科技有限公司", "广州市联发电子有限公司", "北京中天贸易有限公司",
    "上海汇金实业有限公司", "杭州云帆网络科技有限公司", "成都天府智造有限公司",
    "武汉长江光电有限公司", "南京金陵建筑工程有限公司", "苏州工业园区建设发展有限公司",
    "佛山市顺德家具有限公司", "东莞市乐迪电子厂", "珠海格力精密模具有限公司",
    "厦门鹭岛食品有限公司", "青岛海晟物流有限公司", "天津渤海化工有限公司",
    "重庆山城机械制造有限公司", "西安星辰软件开发有限公司", "郑州市中原商贸有限公司",
    "长沙市湘江服饰有限公司", "合肥市庐州汽车配件有限公司", "宁波甬港进出口有限公司",
    "无锡太湖机床有限公司", "常州龙城纺织有限公司", "济南泉城医疗器械有限公司",
    "福州市闽江茶叶有限公司", "石家庄燕赵药业有限公司", "哈尔滨松花江乳业有限公司",
    "沈阳市重工设备有限公司", "太原晋阳煤炭销售有限公司", "昆明春城花卉有限公司",
]

_BANKS = [
    "中国工商银行", "中国农业银行", "中国银行", "中国建设银行", "交通银行",
    "招商银行", "浦发银行", "中信银行", "兴业银行", "民生银行",
    "平安银行", "中国光大银行", "华夏银行", "广发银行",
]


def rand_name():
    return random.choice(_SURNAMES) + random.choice(_GIVEN) + (
        random.choice(_GIVEN) if random.random() < 0.4 else "")


def rand_company():
    return random.choice(_COMPANIES)


def rand_date(year_range=(2024, 2026)):
    y = random.randint(*year_range)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def rand_amount(lo=50, hi=500000):
    """随机金额，保留两位小数"""
    return round(random.uniform(lo, hi), 2)


def rand_serial():
    return random.randint(1, 999)


def rand_account():
    """随机银行账号样式"""
    groups = []
    for _ in range(4):
        groups.append(f"{random.randint(0, 9999):04d}")
    return " ".join(groups)


def rand_flow_no():
    """随机回单流水号"""
    return str(random.randint(10**17, 10**18 - 1))


# -------------------- 图像退化 --------------------

def make_canvas(w, h, paper_tone=245):
    """白底画布，带轻微纸张色差"""
    base = random.randint(-8, 8)
    gray = min(255, max(200, paper_tone + base))
    return Image.new("RGB", (w, h), (gray, gray, gray))


def rotate_slight(img, max_deg=1.2):
    """轻微整体旋转，模拟扫描/拍照"""
    deg = random.uniform(-max_deg, max_deg)
    if abs(deg) < 0.1:
        return img
    return img.rotate(deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def add_noise(img, strength=6):
    """加轻量颗粒噪声"""
    if random.random() > 0.7:
        return img
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * 0.002)):
        x = random.randrange(w)
        y = random.randrange(h)
        dx = random.randint(-strength, strength)
        r, g, b = px[x, y]
        px[x, y] = (max(0, min(255, r + dx)),
                    max(0, min(255, g + dx)),
                    max(0, min(255, b + dx)))
    return img


def add_vignette(img):
    """轻微边缘压暗，模拟扫描件"""
    if random.random() > 0.5:
        return img
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.92, 0.98))


def degrade(img):
    """按随机概率组合各类退化，输出最终图"""
    img = rotate_slight(img)
    img = add_noise(img)
    img = add_vignette(img)
    return img


# -------------------- 文本绘制工具 --------------------

def _text_w(draw, text, fnt):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0]


def center_text(draw, cx, cy, text, fnt, fill=(20, 20, 20)):
    """水平居中绘制"""
    w = _text_w(draw, text, fnt)
    draw.text((cx - w / 2, cy), text, font=fnt, fill=fill)


def cell_text(draw, x0, x1, cy, text, fnt, fill=(20, 20, 20)):
    """单元格内水平居中，垂直按基线对齐"""
    w = _text_w(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - w) / 2, cy), text, font=fnt, fill=fill)


def fit_lines(draw, text, fnt, max_width):
    """把 text 按宽度拆成多行（贪心逐字），返回行列表"""
    lines = []
    cur = ""
    for ch in text:
        if _text_w(draw, cur + ch, fnt) <= max_width:
            cur += ch
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines or [""]


def multiline_cell(draw, x0, x1, cy_top, height, text, fnt, fill=(20, 20, 20),
                   max_lines=2, line_gap=6):
    """单元格内多行居中文本，最多 max_lines 行。cy_top 为单元格顶部基线，height 为单元格高。"""
    lines = fit_lines(draw, text, fnt, x1 - x0 - 8)[:max_lines]
    line_h = _text_w(draw, "国", fnt) + line_gap
    total_h = len(lines) * line_h
    y = cy_top + (height - total_h) / 2
    for ln in lines:
        w = _text_w(draw, ln, fnt)
        draw.text((x0 + (x1 - x0 - w) / 2, y), ln, font=fnt, fill=fill)
        y += line_h
