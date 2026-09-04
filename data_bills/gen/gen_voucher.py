# -*- coding: utf-8 -*-
"""
记账凭证合成生成器
==================
生成通用记账凭证（及收/付/转账凭证）图片 + 字段 dict。
关键特性：
- 会计分录模板库 + 程序保证借贷平衡（借方合计 = 贷方合计）
- 销售类含增值税 13% 价税分离
- 多种版式变体：字号前缀、行数、公司明细、日期、大写金额
"""
import random
import os
from PIL import Image, ImageDraw, ImageFilter
import common

# -------------------- 会计分录模板 --------------------
# debit/credit 为科目，{comp} 占位符会替换为随机往来单位
ENTRY_TEMPLATES = [
    {"summary": "支付货款", "debit": ["应付账款——{comp}"], "credit": ["银行存款"]},
    {"summary": "报销差旅费", "debit": ["管理费用——差旅费"], "credit": ["库存现金"]},
    {"summary": "收到货款", "debit": ["银行存款"], "credit": ["应收账款——{comp}"]},
    {"summary": "销售商品", "debit": ["银行存款"],
     "credit": ["主营业务收入", "应交税费——应交增值税（销项税额）"], "tax": 0.13},
    {"summary": "购买办公用品", "debit": ["管理费用——办公费"], "credit": ["库存现金"]},
    {"summary": "计提工资", "debit": ["管理费用——职工薪酬"], "credit": ["应付职工薪酬"]},
    {"summary": "发放工资", "debit": ["应付职工薪酬"], "credit": ["银行存款"]},
    {"summary": "缴纳增值税", "debit": ["应交税费——未交增值税"], "credit": ["银行存款"]},
    {"summary": "支付房屋租金", "debit": ["管理费用——租赁费"], "credit": ["银行存款"]},
    {"summary": "采购原材料", "debit": ["原材料"], "credit": ["应付账款——{comp}"]},
    {"summary": "借入短期借款", "debit": ["银行存款"], "credit": ["短期借款"]},
    {"summary": "偿还短期借款", "debit": ["短期借款"], "credit": ["银行存款"]},
    {"summary": "收到股东投资", "debit": ["银行存款"], "credit": ["实收资本"]},
    {"summary": "计提固定资产折旧", "debit": ["管理费用——折旧费"], "credit": ["累计折旧"]},
    {"summary": "结转销售成本", "debit": ["主营业务成本"], "credit": ["库存商品"]},
    {"summary": "支付水电费", "debit": ["管理费用——水电费"], "credit": ["银行存款"]},
    {"summary": "销售商品收款", "debit": ["库存现金"],
     "credit": ["主营业务收入", "应交税费——应交增值税（销项税额）"], "tax": 0.13},
    {"summary": "支付运输费", "debit": ["销售费用——运输费"], "credit": ["银行存款"]},
]

# 凭证字号前缀 → 凭证名称
KIND_PREFIX = {
    "记": "记账凭证",
    "收": "收款凭证",
    "付": "付款凭证",
    "转": "转账凭证",
}


def split_amount(total, n):
    """把 total 拆成 n 个和为 total 的金额（保留两位小数）"""
    if n <= 1:
        return [round(total, 2)]
    cents = sorted(random.randint(0, int(round(total * 100))) for _ in range(n - 1))
    bounds = [0] + cents + [int(round(total * 100))]
    parts = [round((bounds[i + 1] - bounds[i]) / 100, 2) for i in range(n)]
    diff = round(round(total, 2) - sum(parts), 2)
    parts[-1] = round(parts[-1] + diff, 2)
    return parts


def build_entry(template):
    """由模板生成借贷记录，返回 (rows, debit_total, credit_total)"""
    comp = common.rand_company() if "{comp}" in str(template.get("debit") + template.get("credit")) else None
    total = common.rand_amount(100, 80000)

    debit_accounts = [a.replace("{comp}", comp) if comp else a for a in template["debit"]]
    credit_accounts = [a.replace("{comp}", comp) if comp else a for a in template["credit"]]

    if template.get("tax"):
        # 价税分离：收入 = total, 税额 = total*rate, 借方 = 收入+税额
        tax = round(total * template["tax"], 2)
        debit_amounts = [round(total + tax, 2)]
        credit_amounts = [round(total, 2), round(tax, 2)]
    else:
        debit_amounts = split_amount(total, len(debit_accounts))
        credit_amounts = split_amount(total, len(credit_accounts))

    rows = []
    for subj, amt in zip(debit_accounts, debit_amounts):
        rows.append({"summary": template["summary"], "subject": subj, "debit": amt, "credit": None})
    for subj, amt in zip(credit_accounts, credit_amounts):
        rows.append({"summary": template["summary"], "subject": subj, "debit": None, "credit": amt})

    debit_total = round(sum(r["debit"] or 0 for r in rows), 2)
    credit_total = round(sum(r["credit"] or 0 for r in rows), 2)
    assert abs(debit_total - credit_total) < 0.01, "借贷不平衡！"
    return rows, debit_total, credit_total


def generate(seed=None, handwritten=False, anomaly=None):
    """生成一张记账凭证，返回 (PIL.Image, fields dict)。handwritten=True 时正文用楷体模拟手写。"""
    if seed is not None:
        random.seed(seed)

    prefix = random.choice(list(KIND_PREFIX.keys()))
    kind = KIND_PREFIX[prefix]
    serial = common.rand_serial()
    date = common.rand_date()
    attach = random.randint(0, 12)
    template = random.choice(ENTRY_TEMPLATES)
    rows, debit_total, credit_total = build_entry(template)

    # ---------------- 内容异常（审计不合规场景，label=1） ----------------
    anomaly_note = None
    upper_display = None
    if anomaly == "future_date":
        date = f"{random.randint(2030, 2099)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        anomaly_note = f"凭证日期为未来日期（{date}），不合规"
    elif anomaly == "upper_mismatch":
        wrong = round(debit_total * random.uniform(1.1, 3.0), 2)
        upper_display = common.rmb_upper(wrong)
        anomaly_note = f"金额大写写为{upper_display}，与数字合计{debit_total:,.2f}不一致"
    elif anomaly == "imbalance":
        last = rows[-1]
        delta = round(common.rand_amount(60, 800), 2)
        if last["credit"] is not None:
            last["credit"] = round(last["credit"] + delta, 2)
        else:
            last["debit"] = round(last["debit"] + delta, 2)
        debit_total = round(sum(r["debit"] or 0 for r in rows), 2)
        credit_total = round(sum(r["credit"] or 0 for r in rows), 2)
        anomaly_note = f"借方合计{debit_total:,.2f}与贷方合计{credit_total:,.2f}不平衡"
    elif anomaly == "missing_serial":
        serial = None
        anomaly_note = "凭证字号缺失"
    elif anomaly == "alteration":
        anomaly_note = "金额栏存在涂改痕迹"

    maker = common.rand_name()
    auditor = common.rand_name()
    bookkeeper = common.rand_name()
    cashier = common.rand_name()

    fields = {
        "kind": kind,
        "prefix": prefix,
        "serial": serial,
        "date": date,
        "attach": attach,
        "rows": rows,
        "debit_total": debit_total,
        "credit_total": credit_total,
        "maker": maker,
        "auditor": auditor,
        "bookkeeper": bookkeeper,
        "cashier": cashier,
        "anomaly_note": anomaly_note,
    }

    # ---------------- 绘制 ----------------
    img_w, img_h = 1200, 620
    img = common.make_canvas(img_w, img_h)
    draw = ImageDraw.Draw(img)

    ink = (25, 25, 25)
    f_title = common.font("hei", 44)
    f_head = common.font("song", 26)
    f_cell = common.font("kai" if handwritten else "song", 26)
    f_small = common.font("fang", 22)

    # 标题
    common.center_text(draw, img_w / 2, 34, kind, f_title, fill=ink)
    # 字号 + 日期
    if serial is not None:
        common.center_text(draw, 150, 104, f"{prefix}字第{serial:04d}号", f_head, fill=ink)
    common.center_text(draw, img_w - 160, 104, date, f_head, fill=ink)
    draw.text((90, 142), f"附单据 {attach} 张", font=f_small, fill=ink)

    # 表格几何
    x0, x1 = 80, 1120
    col_ws = [300, 340, 200, 200]  # 摘要 / 会计科目 / 借方金额 / 贷方金额
    xs = []
    cur = x0
    for w in col_ws:
        xs.append(cur)
        cur += w
    xs.append(x1)
    y0 = 170
    h_head, h_row, h_total = 50, 58, 50
    n = len(rows)
    table_bottom = y0 + h_head + h_row * n + h_total

    # 画网格线
    for x in xs:
        draw.line([(x, y0), (x, table_bottom)], fill=ink, width=2)
    ys = [y0]
    y = y0
    draw.line([(x0, y), (x1, y)], fill=ink, width=2)
    y += h_head
    for _ in range(n):
        draw.line([(x0, y), (x1, y)], fill=ink, width=2)
        y += h_row
    draw.line([(x0, y), (x1, y)], fill=ink, width=2)
    draw.line([(x0, table_bottom), (x1, table_bottom)], fill=ink, width=2)

    # 表头
    headers = ["摘要", "会计科目", "借方金额", "贷方金额"]
    for i, htext in enumerate(headers):
        cy = y0 + (h_head - 32) / 2
        common.cell_text(draw, xs[i], xs[i + 1], cy, htext, f_head, fill=ink)

    # 数据行
    def draw_row(row, ry, bottom):
        mid = ry + (bottom - ry - 32) / 2
        common.cell_text(draw, xs[0], xs[1], mid, row["summary"], f_cell, fill=ink)
        common.multiline_cell(draw, xs[1], xs[2], ry, bottom - ry, row["subject"], f_cell, fill=ink, max_lines=2)
        # 金额右对齐
        for col, val in ((2, row["debit"]), (3, row["credit"])):
            if val is not None:
                txt = common.money_lower(val)
                w = common._text_w(draw, txt, f_cell)
                draw.text((xs[col + 1] - w - 16, mid), txt, font=f_cell, fill=ink)

    for i, row in enumerate(rows):
        ry = y0 + h_head + i * h_row
        draw_row(row, ry, ry + h_row)

    # 合计行
    yt = y0 + h_head + h_row * n
    cy = yt + (h_total - 32) / 2
    common.cell_text(draw, xs[0], xs[2], cy, "合  计", f_head, fill=ink)
    for col, val in ((2, debit_total), (3, credit_total)):
        txt = common.money_lower(val)
        w = common._text_w(draw, txt, f_head)
        draw.text((xs[col + 1] - w - 16, cy), txt, font=f_head, fill=ink)

    # 页脚：大写金额 + 签名
    fy = table_bottom + 26
    upper = upper_display if upper_display else common.rmb_upper(debit_total)
    draw.text((x0, fy), f"人民币（大写）：{upper}", font=f_small, fill=ink)

    fy2 = fy + 36
    sigs = [
        ("制单", maker),
        ("审核", auditor),
        ("记账", bookkeeper),
        ("出纳", cashier),
    ]
    sig_x = [x0, x0 + 280, x0 + 560, x0 + 840]
    for (label, name), sx in zip(sigs, sig_x):
        draw.text((sx, fy2), f"{label}：{name}", font=f_small, fill=ink)

    if anomaly == "alteration":
        import degrade
        img = degrade.draw_ink_alteration(img, xs[2], y0 + h_head, xs[4], table_bottom)

    img = common.degrade(img)
    if handwritten:
        img = common.add_noise(img, strength=10)
    return img, fields


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "_preview")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(5):
        img, f = generate(seed=100 + i)
        img.save(os.path.join(out_dir, f"voucher_preview_{i}.png"))
        print(f"[{i}] {f['kind']} {f['prefix']}字第{f['serial']:04d}号 "
              f"{f['date']} 借:{f['debit_total']:.2f} 贷:{f['credit_total']:.2f} "
              f"行数={len(f['rows'])}")
    print("预览已保存到 _preview/")
