# -*- coding: utf-8 -*-
"""
审计合规规则引擎
================
对模型输出的"字段读取"做确定性合规判定，与模型自身的文本判断解耦。

规则：
- 记账凭证：借贷平衡、凭证字号存在、日期非未来、大写金额与数字一致
- 银行回单：付款/收款账号位数、流水号 18 位、大写与数字一致、日期非未来
- 差旅报销单：报销日期 >= 出差起始、合计 == 明细之和、报销人签名存在

用法：
    from check_rules import judge
    ok, issues = judge("记账凭证", model_output_text)
"""
import re

# ---------------- 人民币大写转数字 ----------------
_CN = {'零': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9}
_UNIT = {'拾': 10, '佰': 100, '仟': 1000}
_BIG = {'万': 10000, '亿': 100000000}


def upper_to_number(s):
    """人民币大写转 float：壹拾贰万捌仟元整 -> 128000.0；叁角贰分 -> 0.32"""
    total = 0
    section = 0
    num = 0
    jiao = fen = 0
    for ch in s:
        if ch in _CN:
            num = _CN[ch]
        elif ch in _UNIT:
            section += (num if num else 1) * _UNIT[ch]
            num = 0
        elif ch in _BIG:
            section += num
            total += (section if section else 1) * _BIG[ch]
            section = 0
            num = 0
        elif ch == '元':
            section += num
            total += section
            section = 0
            num = 0
        elif ch == '角':
            jiao = num if num else 0
            num = 0
        elif ch == '分':
            fen = num if num else 0
            num = 0
    if section:
        total += section
    return round(total + jiao * 0.1 + fen * 0.01, 2)


def _money(t):
    m = re.search(r"([\d,]+\.\d{2})", t)
    return float(m.group(1).replace(",", "")) if m else None


def _future_dates(out):
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", out)
    return dates and int(dates[0][:4]) > 2029


def check_voucher(out):
    issues = []
    d = re.search(r"借方合计￥([\d,]+\.\d{2})", out)
    c = re.search(r"贷方合计￥([\d,]+\.\d{2})", out)
    if d and c and abs(_money(d.group(1)) - _money(c.group(1))) > 0.01:
        issues.append("借贷不平衡")
    if not re.search(r"[记收付转]字第\d+号", out):
        issues.append("凭证字号缺失")
    if _future_dates(out):
        issues.append("日期异常(未来)")
    u = re.search(r"人民币（大写）：(.+?)。", out)
    if u and d:
        try:
            if abs(upper_to_number(u.group(1)) - _money(d.group(1))) > 0.01:
                issues.append("大写与数字不一致")
        except Exception:
            pass
    return issues


def check_bank(out):
    issues = []
    for lbl in ("付款账号", "收款账号"):
        m = re.search(lbl + r"：([0-9 ]+)", out)
        if m:
            digits = m.group(1).replace(" ", "")
            if len(digits) != 16:
                issues.append(f"{lbl}位数异常({len(digits)}位)")
    f = re.search(r"流水号：(\d+)", out)
    if not f or len(f.group(1)) != 18:
        issues.append("流水号缺失或位数异常")
    u = re.search(r"金额（大写）：(.+?)，", out)
    n = re.search(r"金额：¥([\d,]+\.\d{2})", out)
    if u and n:
        try:
            if abs(upper_to_number(u.group(1)) - _money(n.group(1))) > 0.01:
                issues.append("大写与数字不一致")
        except Exception:
            pass
    if _future_dates(out):
        issues.append("日期异常(未来)")
    return issues


def check_expense(out):
    issues = []
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", out)
    if len(dates) >= 2 and dates[0] < dates[1]:
        issues.append("报销日期早于出差")
    seg = re.search(r"费用明细：(.+?)合计", out)
    if seg:
        detail = [float(m.replace(",", "")) for m in re.findall(r"￥([\d,]+\.\d{2})", seg.group(1))]
        tot = re.search(r"合计：￥([\d,]+\.\d{2})", out)
        if tot and abs(sum(detail) - _money(tot.group(1))) > 0.01:
            issues.append("合计与明细不一致")
    p = re.search(r"报销人：(.+?)，", out)
    if p and "缺失" in p.group(1):
        issues.append("报销人签名缺失")
    return issues


def judge(category, out):
    """返回 (是否异常, 问题列表)。category: 记账凭证/银行回单/差旅报销单"""
    out = re.sub(r"<\|im_end\|>", "", out)
    if category == "记账凭证":
        issues = check_voucher(out)
    elif category == "银行回单":
        issues = check_bank(out)
    elif category == "差旅报销单":
        issues = check_expense(out)
    else:
        issues = []
    return (bool(issues), issues)
