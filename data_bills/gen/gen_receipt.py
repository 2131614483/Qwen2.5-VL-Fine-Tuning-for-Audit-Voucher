# -*- coding: utf-8 -*-
"""
回单 / 报销单合成生成器
======================
- 银行电子回单：多银行版式、收付方/账号/金额大写/摘要/流水号
- 差旅报销单：明细表（日期/摘要/金额）+ 合计 + 审批签字
返回 (PIL.Image, fields dict)
"""
import random
from PIL import Image, ImageDraw, ImageFilter
import common


def _draw_hr(draw, x0, x1, y, color=(200, 200, 200)):
    draw.line([(x0, y), (x1, y)], fill=color, width=1)


# ==================== 银行电子回单 ====================

_SUMMARIES = ["货款", "服务费", "咨询费", "差旅费报销", "办公用品", "劳务费", "租金", "工程款", "工资", "预付款"]
_PURPOSES = ["货款结算", "日常采购", "费用报销", "合同付款", "劳务报酬", "工资发放", "利息支出", "货款"]


def gen_bank_receipt(seed=None, anomaly=None):
    if seed is not None:
        random.seed(seed)
    bank = random.choice(common._BANKS)
    payer, payee = common.rand_company(), common.rand_company()
    while payer == payee:
        payee = common.rand_company()
    amount = common.rand_amount(500, 500000)
    summary = random.choice(_SUMMARIES)
    flow_no = common.rand_flow_no()
    fields = {
        "kind": "银行回单",
        "bank": bank,
        "txn_date": common.rand_date(),
        "txn_time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
        "payer": payer,
        "payer_acct": common.rand_account(),
        "payee": payee,
        "payee_acct": common.rand_account(),
        "amount": round(amount, 2),
        "upper": common.rmb_upper(amount),
        "summary": summary,
        "flow_no": flow_no,
    }

    # 内容异常（审计不合规场景，label=1）
    anomaly_note = None
    if anomaly == "upper_mismatch":
        wrong = round(amount * random.uniform(1.1, 3.0), 2)
        fields["upper"] = common.rmb_upper(wrong)
        anomaly_note = f"金额大写写为{fields['upper']}，与数字金额{amount:,.2f}不一致"
    elif anomaly == "missing_flow":
        fields["flow_no"] = None
        anomaly_note = "流水号缺失"
    elif anomaly == "bad_account":
        fields["payer_acct"] = fields["payer_acct"].replace(" ", "")[: random.randint(8, 15)]
        anomaly_note = "付款账号位数异常，疑似录入错误"
    elif anomaly == "alteration":
        anomaly_note = "金额栏存在涂改痕迹"
    fields["anomaly_note"] = anomaly_note

    img = common.make_canvas(1000, 640)
    draw = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    gray = (90, 90, 90)
    red = (180, 30, 30)
    f_title = common.font("hei", 34)
    f_label = common.font("song", 24)
    f_value = common.font("song", 26)
    f_amount = common.font("hei", 34)
    f_small = common.font("song", 18)

    # 标题（银行名 + 电子回单）
    common.center_text(draw, 500, 34, bank, f_title, fill=ink)
    common.center_text(draw, 500, 78, "电 子 回 单", f_title, fill=ink)
    # 顶部装饰线
    draw.rectangle([(80, 122), (920, 124)], fill=(180, 30, 30))

    # 字段区
    x0, x1 = 110, 890
    rows = [
        ("交易日期", fields["txn_date"], "交易时间", fields["txn_time"]),
        ("付款方", fields["payer"], None, None),
        ("付款账号", fields["payer_acct"], None, None),
        ("收款方", fields["payee"], None, None),
        ("收款账号", fields["payee_acct"], None, None),
    ]
    y = 160
    lh = 48
    for item in rows:
        lab, val, lab2, val2 = item
        draw.text((x0, y), f"{lab}：", font=f_label, fill=gray)
        draw.text((x0 + 110, y), val, font=f_value, fill=ink)
        if lab2:
            draw.text((x0 + 430, y), f"{lab2}：", font=f_label, fill=gray)
            draw.text((x0 + 540, y), val2, font=f_value, fill=ink)
        _draw_hr(draw, x0, x1, y + lh - 4)
        y += lh

    # 金额（大写 + 小写），突出显示
    y += 6
    amount_y = y
    draw.text((x0, y), "金额（大写）：", font=f_label, fill=gray)
    draw.text((x0 + 150, y), fields["upper"], font=f_value, fill=ink)
    y += lh
    draw.text((x0, y), "金额：", font=f_label, fill=gray)
    draw.text((x0 + 110, y), f"¥{common.money_lower(fields['amount'])}", font=f_amount, fill=red)
    _draw_hr(draw, x0, x1, y + lh - 4)
    y += lh

    # 摘要 + 流水号
    draw.text((x0, y), "摘  要：", font=f_label, fill=gray)
    draw.text((x0 + 110, y), fields["summary"], font=f_value, fill=ink)
    if fields["flow_no"]:
        draw.text((x0 + 430, y), "流水号：", font=f_label, fill=gray)
        draw.text((x0 + 540, y), fields["flow_no"], font=f_small, fill=ink)
    y += lh

    if anomaly == "alteration":
        import degrade
        img = degrade.draw_ink_alteration(img, x0, int(amount_y), x1, int(amount_y) + lh * 2)

    # 底部提示
    draw.text((x0, 580), "（银行电子回单仅供查询，不代替银行交易凭证）", font=f_small, fill=gray)

    return common.degrade(img), fields


# ==================== 差旅报销单 ====================

_DEPTS = ["技术部", "销售部", "市场部", "财务部", "人事部", "研发部", "行政部", "采购部", "生产部"]
_TRIPS = ["北京出差", "上海出差", "广州出差", "深圳出差", "成都出差", "杭州出差", "武汉出差", "西安出差", "南京出差", "长沙出差", "郑州出差", "重庆出差"]
_EXPENSE_ITEMS = ["交通费", "住宿费", "餐饮费", "市内交通费", "通讯费", "材料费", "资料费", "住宿费（酒店）", "高铁票", "飞机票"]


def gen_expense_report(seed=None, handwritten=False, anomaly=None):
    if seed is not None:
        random.seed(seed)
    import datetime
    start_dt = common.rand_date()
    y0_, m0_, d0_ = (int(x) for x in start_dt.split("-"))
    n = random.randint(2, 5)
    days = [datetime.date(y0_, m0_, d0_) + datetime.timedelta(days=i) for i in range(n)]
    end_dt = days[-1].strftime("%Y-%m-%d")
    import datetime as _dt
    date_str = (days[-1] + _dt.timedelta(days=random.randint(1, 15))).strftime("%Y-%m-%d")
    reason = random.choice(_TRIPS)
    items = random.sample(_EXPENSE_ITEMS, n)
    amounts = [common.rand_amount(60, 1500) for _ in range(n)]
    total = round(sum(amounts), 2)
    fields = {
        "kind": "差旅报销单",
        "department": random.choice(_DEPTS),
        "date": date_str,
        "start_date": start_dt,
        "end_date": end_dt,
        "reason": reason,
        "rows": [{"date": d.strftime("%Y-%m-%d"), "item": items[i], "amount": round(amounts[i], 2)}
                 for i, d in enumerate(days)],
        "total": total,
        "upper": common.rmb_upper(total),
        "employee": common.rand_name(),
        "manager": common.rand_name(),
        "finance": common.rand_name(),
    }

    # 内容异常（审计不合规场景，label=1）
    anomaly_note = None
    if anomaly == "total_mismatch":
        delta = round(common.rand_amount(50, 600), 2)
        fields["total"] = round(total + delta, 2)
        fields["upper"] = common.rmb_upper(fields["total"])
        anomaly_note = f"合计金额{fields['total']:,.2f}与明细金额合计{total:,.2f}不一致"
    elif anomaly == "missing_signature":
        fields["employee"] = None
        anomaly_note = "报销人签名缺失"
    elif anomaly == "late_trip":
        fields["date"] = "2020-01-01"
        anomaly_note = f"报销日期{fields['date']}早于出差起始{fields['start_date']}"
    elif anomaly == "alteration":
        anomaly_note = "金额栏存在涂改痕迹"
    fields["anomaly_note"] = anomaly_note

    img = common.make_canvas(1000, 580)
    draw = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    f_title = common.font("hei", 36)
    f_head = common.font("song", 24)
    f_cell = common.font("kai" if handwritten else "song", 24)
    f_small = common.font("kai" if handwritten else "fang", 20)

    common.center_text(draw, 500, 30, "差 旅 费 报 销 单", f_title, fill=ink)

    # 顶部信息行
    draw.text((90, 90), f"报销部门：{fields['department']}", font=f_head, fill=ink)
    draw.text((560, 90), f"报销日期：{fields['date']}", font=f_head, fill=ink)
    draw.text((90, 126), f"出差区间：{fields['start_date']} 至 {fields['end_date']}", font=f_head, fill=ink)
    draw.text((560, 126), f"事由：{fields['reason']}", font=f_head, fill=ink)

    # 表格
    x0, x1 = 90, 910
    col_ws = [180, 300, 180, 160]
    xs = []
    cur = x0
    for w in col_ws:
        xs.append(cur)
        cur += w
    xs.append(x1)
    y0 = 170
    h_head, h_row, h_total = 44, 48, 48
    n_rows = len(fields["rows"])
    bottom = y0 + h_head + h_row * n_rows + h_total

    for x in xs:
        draw.line([(x, y0), (x, bottom)], fill=ink, width=2)
    y = y0
    draw.line([(x0, y), (x1, y)], fill=ink, width=2)
    y += h_head
    for _ in range(n_rows):
        draw.line([(x0, y), (x1, y)], fill=ink, width=2)
        y += h_row
    draw.line([(x0, y), (x1, y)], fill=ink, width=2)
    draw.line([(x0, bottom), (x1, bottom)], fill=ink, width=2)

    headers = ["日期", "费用摘要", "金额", "备注"]
    for i, htext in enumerate(headers):
        common.cell_text(draw, xs[i], xs[i + 1], y0 + (h_head - 30) / 2, htext, f_head, fill=ink)

    for i, r in enumerate(fields["rows"]):
        ry = y0 + h_head + i * h_row
        mid = ry + (h_row - 30) / 2
        common.cell_text(draw, xs[0], xs[1], mid, r["date"], f_cell, fill=ink)
        common.cell_text(draw, xs[1], xs[2], mid, r["item"], f_cell, fill=ink)
        txt = common.money_lower(r["amount"])
        w = common._text_w(draw, txt, f_cell)
        draw.text((xs[3] - w - 12, mid), txt, font=f_cell, fill=ink)
        common.cell_text(draw, xs[3], xs[4], mid, "", f_cell, fill=ink)

    # 合计
    yt = y0 + h_head + h_row * n_rows
    mid = yt + (h_total - 30) / 2
    common.cell_text(draw, xs[0], xs[2], mid, "合  计", f_head, fill=ink)
    txt = common.money_lower(fields["total"])
    w = common._text_w(draw, txt, f_head)
    draw.text((xs[3] - w - 12, mid), txt, font=f_head, fill=ink)

    # 页脚：大写金额 + 签名
    fy = bottom + 24
    draw.text((x0, fy), f"人民币（大写）：{fields['upper']}", font=f_small, fill=ink)
    fy2 = fy + 34
    sigs = [("报销人", fields["employee"]), ("部门经理", fields["manager"]), ("财务审核", fields["finance"])]
    sig_x = [x0, x0 + 300, x0 + 600]
    for (label, name), sx in zip(sigs, sig_x):
        if name:
            draw.text((sx, fy2), f"{label}：{name}", font=f_small, fill=ink)

    if anomaly == "alteration":
        import degrade
        img = degrade.draw_ink_alteration(img, xs[2], yt, xs[4], bottom)

    return common.degrade(img), fields


def generate(kind="receipt", seed=None, handwritten=False, anomaly=None):
    """统一入口。kind: 'receipt' 银行回单 / 'expense' 差旅报销单"""
    if kind == "receipt":
        return gen_bank_receipt(seed, anomaly=anomaly)
    return gen_expense_report(seed, handwritten=handwritten, anomaly=anomaly)


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "_preview")
    os.makedirs(out, exist_ok=True)
    for i in range(3):
        img, f = gen_bank_receipt(seed=200 + i)
        img.save(os.path.join(out, f"receipt_preview_{i}.png"))
        print(f"回单[{i}] {f['bank']} {f['amount']:.2f} {f['payer']} -> {f['payee']}")
    for i in range(3):
        img, f = gen_expense_report(seed=300 + i)
        img.save(os.path.join(out, f"expense_preview_{i}.png"))
        print(f"报销单[{i}] {f['department']} {f['total']:.2f} {f['reason']}")
    print("预览已保存")
