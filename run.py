import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pymysql


DB_HOST = os.getenv("TASK2_DB_HOST", "172.16.48.27")
DB_PORT = int(os.getenv("TASK2_DB_PORT", "3306"))
DB_USER = os.getenv("TASK2_DB_USER", "test_user")
DB_PASSWORD = os.getenv("TASK2_DB_PASSWORD", "R6#pV9@kT3!xM2$q")
DB_NAME = os.getenv("TASK2_DB_NAME", "cmb_contest")
BASE_TABLE = os.getenv("TASK2_BASE_TABLE", "train_base_table")
ACTION_TABLE = os.getenv("TASK2_ACTION_TABLE", "train_action_table")

CURRENT_YEAR = 2025
CURRENT_MONTH = 3
DEFAULT_INFLATION = 0.02
DEFAULT_RETURN = 0.02
DEFAULT_LIFE_EXPECTANCY = 80

PRODUCTS = {
    "现金理财": {"risk": 1, "return": 0.015},
    "定期存款": {"risk": 1, "return": 0.020},
    "短债类产品": {"risk": 2, "return": 0.024},
    "固收+产品": {"risk": 3, "return": 0.0425},
    "权益类产品": {"risk": 4, "return": 0.060},
    "年金险": {"risk": 1, "return": 0.025},
}

PRODUCT_PRIORITY = ["现金理财", "定期存款", "短债类产品", "固收+产品", "权益类产品", "年金险"]


def _safe_table(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        raise ValueError("invalid table name")
    return f"`{name}`"


BASE_SQL_TABLE = _safe_table(BASE_TABLE)
ACTION_SQL_TABLE = _safe_table(ACTION_TABLE)


def _conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
    )


def query_one(sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()
    finally:
        conn.close()


def query_all(sql: str, args: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return list(cur.fetchall())
    finally:
        conn.close()


def money(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("元", "").replace("万", "0000").strip()
    try:
        return float(text)
    except Exception:
        return 0.0


def rnd(value: float) -> int:
    return int(math.floor(value + 0.5))


def pct(value: float) -> int:
    return int(math.ceil(value * 100 - 1e-12))


def fmt_int(value: float) -> str:
    return str(rnd(value))


def extract_user_id(text: str) -> Optional[str]:
    m = re.search(r"V\d{6,}", text, re.I)
    return m.group(0).upper() if m else None


def extract_first_int(text: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else default


def risk_level(rsk_cd: str) -> int:
    m = re.search(r"R(\d)", str(rsk_cd or "R1"))
    return int(m.group(1)) if m else 1


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    sql = f"""
        SELECT User_ID, Age, Gender, Rsk_Cd, Net_Asset, Monthly_Income,
               Monthly_Expend, Pension, Enterprise_Ann
        FROM {BASE_SQL_TABLE}
        WHERE User_ID=%s
        LIMIT 1
    """
    return query_one(sql, (user_id,))


def retirement_months(user: Dict[str, Any]) -> int:
    age = int(money(user.get("Age")))
    gender = str(user.get("Gender") or "")
    base_age = 60 if "男" in gender else 55
    birth_year = CURRENT_YEAR - age
    original_year = birth_year + base_age
    months_since_policy = max(0, (original_year - 2025) * 12 + (CURRENT_MONTH - 1))
    delay = min(36, int(math.ceil(months_since_policy / 4.0)))
    return max(0, (base_age - age) * 12 + delay)


def retirement_age_months(user: Dict[str, Any]) -> int:
    age = int(money(user.get("Age")))
    return age * 12 + retirement_months(user)


def years_months(months: int) -> Tuple[int, int]:
    return months // 12, months % 12


def years_months_text(months: int) -> str:
    y, m = years_months(months)
    if m == 0:
        return f"{y}年"
    return f"{y}年{m}个月"


def monthly_surplus(user: Dict[str, Any]) -> float:
    return money(user.get("Monthly_Income")) - money(user.get("Monthly_Expend"))


def future_monthly_expend(user: Dict[str, Any], inflation: float = DEFAULT_INFLATION,
                          split_after_years: Optional[int] = None,
                          later_inflation: Optional[float] = None) -> float:
    months = retirement_months(user)
    expend = money(user.get("Monthly_Expend"))
    if split_after_years is None or later_inflation is None:
        return expend * ((1 + inflation / 12) ** months)
    first_months = min(months, split_after_years * 12)
    second_months = max(0, months - first_months)
    return expend * ((1 + inflation / 12) ** first_months) * ((1 + later_inflation / 12) ** second_months)


def pv_annuity(payment: float, months: int, discount_rate: float) -> float:
    monthly = discount_rate / 12
    if months <= 0:
        return 0.0
    if abs(monthly) < 1e-12:
        return payment * months
    return sum(payment / ((1 + monthly) ** k) for k in range(months))


def retirement_need(user: Dict[str, Any], life_expectancy: int = DEFAULT_LIFE_EXPECTANCY,
                    inflation: float = DEFAULT_INFLATION,
                    investment_return: float = DEFAULT_RETURN,
                    split_after_years: Optional[int] = None,
                    later_inflation: Optional[float] = None) -> Dict[str, float]:
    retire_age_months = retirement_age_months(user)
    retirement_age_years = retire_age_months / 12
    post_months = max(0, rnd((life_expectancy - retirement_age_years) * 12))
    monthly_exp = future_monthly_expend(user, inflation, split_after_years, later_inflation)
    pension = money(user.get("Pension"))
    enterprise_ann = money(user.get("Enterprise_Ann"))

    retire_inflation = later_inflation if later_inflation is not None else inflation
    if abs(investment_return - retire_inflation) < 1e-12:
        living_need = rnd(monthly_exp) * post_months
    else:
        living_need = sum(monthly_exp * (((1 + retire_inflation / 12) / (1 + investment_return / 12)) ** k)
                          for k in range(post_months))
    pension_pv = pv_annuity(pension, post_months, retire_inflation)
    gap = max(0.0, living_need - pension_pv - enterprise_ann)
    return {
        "post_months": float(post_months),
        "monthly_exp": monthly_exp,
        "living_need": living_need,
        "pension_pv": pension_pv,
        "enterprise_ann": enterprise_ann,
        "gap": gap,
    }


def future_accumulation(user: Dict[str, Any], annual_return: float = DEFAULT_RETURN) -> float:
    months = retirement_months(user)
    monthly = annual_return / 12
    net_asset = money(user.get("Net_Asset"))
    surplus = monthly_surplus(user)
    if months <= 0:
        return net_asset
    if abs(monthly) < 1e-12:
        return net_asset + surplus * months
    factor = (1 + monthly) ** months
    return net_asset * factor + surplus * (factor - 1) / monthly


def allowed_products(rsk_cd: str) -> List[str]:
    level = risk_level(rsk_cd)
    allowed = []
    for name in PRODUCT_PRIORITY:
        if PRODUCTS[name]["risk"] <= level or name == "年金险":
            allowed.append(name)
    return allowed


def highest_return_product(user: Dict[str, Any]) -> str:
    items = allowed_products(str(user.get("Rsk_Cd") or "R1"))
    return max(items, key=lambda name: PRODUCTS[name]["return"])


def product_case_sql() -> str:
    return """
        CASE
            WHEN rsk_lvl IN ('R4','R5') AND prod_typ='基金' THEN '权益类产品'
            WHEN rsk_lvl='R2' AND prod_typ IN ('基金','理财') THEN '短债类产品'
            WHEN rsk_lvl='R3' AND prod_typ IN ('基金','理财') THEN '固收+产品'
            WHEN prod_sub_typ='一般性' AND prod_typ='存款' THEN '定期存款'
            WHEN prod_sub_typ='现金' THEN '现金理财'
            WHEN prod_sub_typ IN ('税延养老年金','养老年金') AND prod_typ='保险' THEN '年金险'
            WHEN prod_sub_typ IN ('税延养老年金','养老年金') THEN '年金险'
            ELSE '其他'
        END
    """


def top_behavior_product(user_id: str, buy_first: bool = False) -> Tuple[str, int]:
    case_sql = product_case_sql()
    action_filter = "AND action_typ='购买'" if buy_first else ""
    sql = f"""
        SELECT {case_sql} AS product, COUNT(*) AS cnt
        FROM {ACTION_SQL_TABLE}
        WHERE user_id=%s AND prod_typ NOT IN ('非财富') {action_filter}
        GROUP BY product
        HAVING product <> '其他'
        ORDER BY cnt DESC,
                 FIELD(product, '现金理财','定期存款','短债类产品','固收+产品','权益类产品','年金险')
        LIMIT 1
    """
    row = query_one(sql, (user_id,))
    if row:
        return str(row.get("product") or "现金理财"), int(row.get("cnt") or 0)
    if buy_first:
        return top_behavior_product(user_id, buy_first=False)
    return "现金理财", 0


def avg_age_for_equity_views(min_count: int) -> int:
    sql = f"""
        WITH T AS (
            SELECT user_id, COUNT(*) AS view_cnt
            FROM {ACTION_SQL_TABLE}
            WHERE action_typ IN ('浏览详情','浏览持仓')
              AND prod_typ='基金'
              AND rsk_lvl IN ('R4','R5')
            GROUP BY user_id
            HAVING view_cnt >= %s
        )
        SELECT AVG(b.Age) AS avg_age
        FROM T INNER JOIN {BASE_SQL_TABLE} b ON b.User_ID = T.user_id
    """
    row = query_one(sql, (min_count,))
    return rnd(money(row.get("avg_age") if row else 0))


def count_age_ge(age: int) -> int:
    sql = f"SELECT COUNT(*) AS cnt FROM {BASE_SQL_TABLE} WHERE Age >= %s"
    row = query_one(sql, (age,))
    return int(row.get("cnt") or 0) if row else 0


def parse_life_expectancy(text: str) -> int:
    m = re.search(r"(?:寿命|人均寿命|预期寿命).*?(\d+)\s*岁", text)
    return int(m.group(1)) if m else DEFAULT_LIFE_EXPECTANCY


def parse_inflation_scenario(text: str) -> Tuple[Optional[int], Optional[float]]:
    m = re.search(r"(\d+)\s*年后.*?通胀率.*?(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2)) / 100


def min_risk_allocation(user: Dict[str, Any]) -> str:
    need = retirement_need(user)
    gap = rnd(need["gap"])
    candidates = []
    for product in allowed_products(str(user.get("Rsk_Cd") or "R1")):
        if product == "年金险":
            continue
        acc = future_accumulation(user, PRODUCTS[product]["return"])
        if acc >= gap:
            candidates.append((PRODUCTS[product]["risk"], PRODUCTS[product]["return"], product, acc))
    if not candidates:
        return "当前风险承受范围内仅靠现有结余较难覆盖养老缺口，建议提高每月结余或适当提升风险承受能力"
    candidates.sort(key=lambda x: (x[0], x[1]))
    product = candidates[0][2]
    return f"不能，需要改为投资{product}"


def low_vol_allocation(user: Dict[str, Any]) -> str:
    need = retirement_need(user)
    gap = rnd(need["gap"])
    candidates = []
    for product in allowed_products(str(user.get("Rsk_Cd") or "R1")):
        if product in ("现金理财", "定期存款", "年金险"):
            continue
        acc = future_accumulation(user, PRODUCTS[product]["return"])
        if acc >= gap:
            candidates.append((PRODUCTS[product]["risk"], PRODUCTS[product]["return"], product, acc))
    if not candidates:
        product = highest_return_product(user)
        return f"{product}配置100%"
    candidates.sort(key=lambda x: (x[0], x[1]))
    product, acc = candidates[0][2], candidates[0][3]
    main_pct = min(100, pct(gap / acc))
    remaining = 100 - main_pct
    cash = min(10, remaining)
    annuity = remaining - cash
    parts = [f"{product}配置{main_pct}%"]
    if cash:
        parts.append(f"现金理财{cash}%")
    if annuity:
        parts.append(f"年金险{annuity}%")
    return "；".join(parts)


def direct_field_answer(text: str, user: Dict[str, Any]) -> Optional[str]:
    if "年龄" in text or "多大" in text:
        return f"{rnd(money(user.get('Age')))}岁"
    if "性别" in text:
        return str(user.get("Gender") or "")
    if "风险" in text and ("评级" in text or "等级" in text):
        return str(user.get("Rsk_Cd") or "")
    if "净资产" in text:
        return f"{fmt_int(money(user.get('Net_Asset')))}元"
    if "月收入" in text:
        return f"{fmt_int(money(user.get('Monthly_Income')))}元"
    if "月支出" in text and "退休" not in text:
        return f"{fmt_int(money(user.get('Monthly_Expend')))}元"
    if "结余" in text:
        return f"{fmt_int(monthly_surplus(user))}元"
    if "企业年金" in text:
        value = money(user.get("Enterprise_Ann"))
        return "无" if value <= 0 else f"{fmt_int(value)}元"
    if ("退休金" in text or "养老金" in text) and not any(word in text for word in ("缺口", "最低", "现值", "需要", "规划")):
        return f"{fmt_int(money(user.get('Pension')))}元"
    return None


def recommendation_report(user_id: str, user: Dict[str, Any], text: str) -> str:
    months = retirement_months(user)
    retire_age_y, retire_age_m = years_months(retirement_age_months(user))
    need = retirement_need(user, parse_life_expectancy(text))
    pref_product, pref_cnt = top_behavior_product(user_id)
    alloc = low_vol_allocation(user)
    surplus = monthly_surplus(user)
    enterprise = money(user.get("Enterprise_Ann"))
    enterprise_text = "无" if enterprise <= 0 else f"{fmt_int(enterprise)}元"
    monthly_exp = rnd(need["monthly_exp"])
    total_need = rnd(need["living_need"])
    pension_pv = rnd(need["pension_pv"])
    gap = rnd(need["gap"])
    return (
        "1. 基本情况\n"
        f"客户ID：{user_id}，年龄：{rnd(money(user.get('Age')))}岁，性别：{user.get('Gender')}，"
        f"风险评级：{user.get('Rsk_Cd')}。当前净资产：{fmt_int(money(user.get('Net_Asset')))}元，"
        f"每月结余：{fmt_int(surplus)}元（月收入{fmt_int(money(user.get('Monthly_Income')))}元，"
        f"月支出{fmt_int(money(user.get('Monthly_Expend')))}元）。每月退休金："
        f"{fmt_int(money(user.get('Pension')))}元，企业年金：{enterprise_text}。\n"
        "2. 基本假设\n"
        f"预期寿命{parse_life_expectancy(text)}岁，长期通胀率2%，默认投资回报率2%，"
        f"退休年龄约{retire_age_y}岁{retire_age_m}个月，距离退休{years_months_text(months)}。\n"
        "3. 养老目标\n"
        f"维持退休后消费水平不下降，退休时每月支出约为{monthly_exp}元。\n"
        "4. 退休后财富需求测算\n"
        f"退休后预计生活资金需求约{total_need}元，养老金现值约{pension_pv}元，"
        f"企业年金可补充{fmt_int(enterprise)}元，仍需通过投资积累覆盖约{gap}元。\n"
        "5. 产品偏好\n"
        f"根据客户行为记录，客户最偏好的产品类型为{pref_product}"
        f"{'，相关行为约' + str(pref_cnt) + '次' if pref_cnt else ''}。\n"
        "6. 资产配置方式与具体方案\n"
        f"建议采用满足养老目标基础上的稳健配置：{alloc}。\n"
        "7. 其他建议\n"
        "建议定期复核收入、支出、风险评级和养老目标；若寿命预期提高，可增加年金险配置以对冲长寿风险。"
    )


def answer_user_question(text: str, user_id: str, user: Dict[str, Any]) -> str:
    direct = direct_field_answer(text, user)
    if direct is not None:
        return direct

    if "建议书" in text or "规划书" in text or "养老规划" in text and "生成" in text:
        return recommendation_report(user_id, user, text)
    if "行为最多" in text or "偏好" in text and "产品" in text:
        return top_behavior_product(user_id)[0]
    if "未来" in text and ("购买" in text or "买" in text):
        return top_behavior_product(user_id, buy_first=True)[0]
    if "寿命" in text and ("增加" in text or "配置" in text or "延长" in text):
        return "年金险"
    if "距离退休" in text or "还有多久" in text and "退休" in text:
        return years_months_text(retirement_months(user))
    if "刚退休" in text or ("退休时" in text and ("每月" in text or "月支出" in text or "支出" in text) and "最低" not in text):
        split_years, later_inf = parse_inflation_scenario(text)
        return f"{fmt_int(future_monthly_expend(user, split_after_years=split_years, later_inflation=later_inf))}元"
    if "最低" in text and ("积攒" in text or "准备" in text or "储备" in text or "需要" in text):
        split_years, later_inf = parse_inflation_scenario(text)
        need = retirement_need(user, parse_life_expectancy(text), split_after_years=split_years, later_inflation=later_inf)
        return f"{fmt_int(need['gap'])}元"
    if "可以积攒" in text or "能积攒" in text or "积攒下" in text or "可积累" in text:
        return f"{fmt_int(future_accumulation(user))}元"
    if "能否" in text or "能不能" in text or "能达成" in text or "不能" in text and "调整" in text:
        need = rnd(retirement_need(user)["gap"])
        acc = rnd(future_accumulation(user))
        if acc >= need:
            return "能达成目标"
        return min_risk_allocation(user)
    if "收益最大" in text or "收益最大化" in text or "追求投资收益" in text:
        return f"{highest_return_product(user)}配置100%"
    if "最小化风险" in text or "风险波动" in text:
        return low_vol_allocation(user)
    return recommendation_report(user_id, user, text)


def run(inf: str) -> str:
    text = (inf or "").strip()
    if not text:
        return ""
    try:
        if "年龄" in text and ("多少客户" in text or "多少个客户" in text or "客户年龄" in text) and ("以上" in text or "及以上" in text or ">=" in text):
            age = extract_first_int(text, 0) or 0
            return f"{count_age_ge(age)}个"
        if "权益类产品" in text and "平均年龄" in text:
            min_count = 2
            m = re.search(r"(\d+)\s*次", text)
            if m:
                min_count = int(m.group(1))
            return f"{avg_age_for_equity_views(min_count)}岁"

        user_id = extract_user_id(text)
        if not user_id:
            return "请提供客户ID"
        user = get_user(user_id)
        if not user:
            return "未查询到客户信息"
        return answer_user_question(text, user_id, user)
    except Exception:
        return "暂时无法回答该问题"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.stdout.write(run(question))


if __name__ == "__main__":
    main()
