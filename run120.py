import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pymysql


DB_HOST = os.getenv("TASK2_DB_HOST", "172.16.48.27")
DB_PORT = int(os.getenv("TASK2_DB_PORT", "3306"))
DB_USER = os.getenv("TASK2_DB_USER", "test_user")
DB_PASSWORD = os.getenv("TASK2_DB_PASSWORD", "R6#pV9@kT3!xM2$q")
DB_NAME = os.getenv("TASK2_DB_NAME", "cmb_contest")
BASE_TABLE = os.getenv("TASK2_BASE_TABLE", "train_base_table")
ACTION_TABLE = os.getenv("TASK2_ACTION_TABLE", "train_action_table")

ONE_API_URL = os.getenv("ONE_API_URL", "https://one-api-other.nowcoder.com/v1/chat/completions")
ONE_API_KEY = os.getenv("ONE_API_KEY", "sk-lInwrXsr5dg5rrVvB5BeE404C03c46F1841f81Aa6b9d5405")
ONE_API_MODEL = os.getenv("ONE_API_MODEL", "qwen3.6-flash")
LLM_TIMEOUT = float(os.getenv("ONE_API_TIMEOUT", "20"))
LLM_ENABLED = bool(ONE_API_URL) and bool(ONE_API_KEY) and ONE_API_KEY != "YOUR_ONE_API_KEY"
LLM_REPORT_POLISH = os.getenv("ONE_API_REPORT_POLISH", "0").lower() in ("1", "true", "yes")
LLM_FALLBACK = os.getenv("ONE_API_FALLBACK", "1").lower() in ("1", "true", "yes")
DEBUG_LOG_ENABLED = os.getenv("TASK2_DEBUG_LOG", "0").lower() in ("1", "true", "yes")
DEBUG_LOG_DIR = os.getenv("TASK2_DEBUG_LOG_DIR", "/tmp/task2_agent_logs")
_DEBUG_LOG_PATH: Optional[str] = None

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
PRODUCT_NAMES = list(PRODUCTS.keys())


# ============================================================
# 数据库连接（单次 run() 内共享，避免反复建连）
# ============================================================
_local = threading.local()
_session_lock = threading.Lock()
_session_memory: Dict[str, Dict[str, Any]] = {}
_last_user_id: Optional[str] = None


def debug_log(event: str, **fields: Any) -> None:
    """调试日志默认关闭；开启后只写唯一文件，绝不污染标准输出。"""
    global _DEBUG_LOG_PATH
    if not DEBUG_LOG_ENABLED:
        return
    try:
        if _DEBUG_LOG_PATH is None:
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            _DEBUG_LOG_PATH = os.path.join(
                DEBUG_LOG_DIR,
                f"run_{os.getpid()}_{time.time_ns()}.log",
            )
        payload = {"ts": time.time(), "event": event, **fields}
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _log_preview(text: Optional[str], limit: int = 120) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def _safe_table(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        raise ValueError("invalid table name")
    return f"`{name}`"


BASE_SQL_TABLE = _safe_table(BASE_TABLE)
ACTION_SQL_TABLE = _safe_table(ACTION_TABLE)


def _get_conn():
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = pymysql.connect(
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
        _local.conn = conn
    return conn


def _close_conn() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def query_one(sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()


def query_all(sql: str, args: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return list(cur.fetchall())


# ============================================================
# 大模型调用（用 urllib，避免新增依赖；失败一律回退到规则结果）
# ============================================================
def llm_chat(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 600,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    purpose: str = "unspecified",
    call_site: str = "llm_chat",
) -> Optional[str]:
    if not LLM_ENABLED:
        debug_log(
            "llm_call_skipped",
            call_site=call_site,
            purpose=purpose,
            reason="LLM_ENABLED is false",
        )
        return None
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    payload = json.dumps(
        {
            "model": ONE_API_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        ONE_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {ONE_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.time()
    debug_log(
        "llm_call_start",
        call_site=call_site,
        purpose=purpose,
        model=ONE_API_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout or LLM_TIMEOUT,
        system_chars=len(system_prompt or ""),
        user_chars=len(user_prompt or ""),
        user_preview=_log_preview(user_prompt),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or LLM_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(body)
        content = obj["choices"][0]["message"]["content"]
        result = (content or "").strip()
        debug_log(
            "llm_call_end",
            call_site=call_site,
            purpose=purpose,
            ok=True,
            elapsed_ms=int((time.time() - started) * 1000),
            response_chars=len(result),
            response_preview=_log_preview(result),
        )
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError, KeyError, IndexError, OSError) as exc:
        debug_log(
            "llm_call_end",
            call_site=call_site,
            purpose=purpose,
            ok=False,
            elapsed_ms=int((time.time() - started) * 1000),
            error=repr(exc),
        )
        return None


INTENT_LABELS = [
    "field",          # 直接字段查询：年龄/性别/月收入等
    "count_age",      # 客户群体年龄统计
    "equity_avg_age", # 浏览权益类客户平均年龄
    "behavior_top",   # 客户最偏好/行为最多的产品类型
    "buy_predict",    # 未来最可能购买
    "longevity",      # 长寿风险下增加什么产品
    "retire_in",      # 距离退休还有多久
    "future_expend",  # 退休时每月支出
    "living_need",    # 退休后总花销/总需求
    "min_reserve",    # 退休时最低需积攒
    "pension_pv",     # 养老金/退休金现值
    "future_acc",     # 退休时可积攒
    "feasibility",    # 能否达成 / 如何调整
    "max_return",     # 最大化投资收益的配置
    "low_vol",        # 最小化风险波动的配置
    "report",         # 养老规划建议书
]


def llm_classify_intent(text: str) -> Optional[str]:
    if not LLM_FALLBACK:
        debug_log(
            "llm_call_skipped",
            call_site="llm_classify_intent",
            purpose="规则路由未命中时，用大模型兜底判断问题意图",
            reason="ONE_API_FALLBACK is false",
            question=text,
        )
        return None
    system = "养老Agent意图标签，只输出一个词，无任何其它字符：" + ",".join(INTENT_LABELS)
    prompt = text
    res = llm_chat(
        prompt,
        system_prompt=system,
        max_tokens=16,
        timeout=8,
        purpose="规则路由未命中时，用大模型兜底判断问题意图",
        call_site="llm_classify_intent",
    )
    if not res:
        debug_log("llm_intent", question=text, result=None)
        return None
    res = res.strip().strip("`'\"").lower()
    for label in INTENT_LABELS:
        if label in res:
            debug_log("llm_intent", question=text, result=label, raw=res)
            return label
    debug_log("llm_intent", question=text, result=None, raw=res)
    return None


# ============================================================
# 数值与字段工具
# ============================================================
def money(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("元", "").strip()
    try:
        if text.endswith("万"):
            return float(text[:-1]) * 10000
        return float(text)
    except Exception:
        return 0.0


def rnd(value: float) -> int:
    return int(math.floor(value + 0.5))


def pct(value: float) -> int:
    return int(math.ceil(value * 100 - 1e-12))


def fmt_int(value: float) -> str:
    return str(rnd(value))


def fmt_money(value: float) -> str:
    return f"{rnd(value)} 元"


def fmt_short_money(value: float) -> str:
    return f"{rnd(value)}元"


def fmt_duration(text: str) -> str:
    return text.replace("年", " 年").replace("个月", " 个月").strip()


def fmt_product_name(name: str) -> str:
    return name.replace("固收+产品", "固收 + 产品")


def fmt_rate(value: float) -> str:
    pct_value = value * 100
    if abs(pct_value - round(pct_value)) < 1e-9:
        return f"{int(round(pct_value))}%"
    return f"{pct_value:.2f}".rstrip("0").rstrip(".") + "%"


def extract_user_id(text: str) -> Optional[str]:
    m = re.search(r"V\d{6,}", text, re.I)
    return m.group(0).upper() if m else None


def extract_first_int(text: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else default


def extract_percent(text: str, keywords: Tuple[str, ...]) -> Optional[float]:
    for keyword in keywords:
        patterns = (
            rf"{keyword}[^0-9%]*(\d+(?:\.\d+)?)\s*%",
            rf"(\d+(?:\.\d+)?)\s*%[^，。；、]*{keyword}",
        )
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return float(m.group(1)) / 100
    return None


def risk_level(rsk_cd: str) -> int:
    m = re.search(r"R(\d)", str(rsk_cd or "R1"))
    return int(m.group(1)) if m else 1


# ============================================================
# 客户与退休测算
# ============================================================
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


def future_monthly_expend(
    user: Dict[str, Any],
    inflation: float = DEFAULT_INFLATION,
    split_after_years: Optional[int] = None,
    later_inflation: Optional[float] = None,
) -> float:
    months = retirement_months(user)
    expend = money(user.get("Monthly_Expend"))
    if split_after_years is None or later_inflation is None:
        return expend * ((1 + inflation / 12) ** months)
    first_months = min(months, split_after_years * 12)
    second_months = max(0, months - first_months)
    return (
        expend
        * ((1 + inflation / 12) ** first_months)
        * ((1 + later_inflation / 12) ** second_months)
    )


def pv_annuity(payment: float, months: int, discount_rate: float) -> float:
    monthly = discount_rate / 12
    if months <= 0:
        return 0.0
    if abs(monthly) < 1e-12:
        return payment * months
    return sum(payment / ((1 + monthly) ** k) for k in range(months))


def retirement_need(
    user: Dict[str, Any],
    life_expectancy: int = DEFAULT_LIFE_EXPECTANCY,
    inflation: float = DEFAULT_INFLATION,
    investment_return: float = DEFAULT_RETURN,
    split_after_years: Optional[int] = None,
    later_inflation: Optional[float] = None,
) -> Dict[str, float]:
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
        ratio = (1 + retire_inflation / 12) / (1 + investment_return / 12)
        living_need = sum(monthly_exp * (ratio ** k) for k in range(post_months))
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


ASSUMPTION_WORDS = ("假如", "假设", "如果", "若", "假定")
OPINION_WORDS = ("认为", "觉得", "想要", "希望", "预期", "预计", "计划", "打算", "目标")


def has_assumption(text: str) -> bool:
    return any(word in text for word in ASSUMPTION_WORDS)


def has_opinion(text: str) -> bool:
    return any(word in text for word in OPINION_WORDS)


def parse_life_expectancy_value(text: str) -> Optional[int]:
    patterns = (
        r"(?:寿命|人均寿命|预期寿命|预计寿命|活到|活至|延长到|延长至).*?(\d{2,3})\s*岁",
        r"(\d{2,3})\s*岁[^，。；、]*(?:寿命|去世|身故)",
    )
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    return None


def parse_scenario_overrides(text: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    life = parse_life_expectancy_value(text)
    if life is not None:
        overrides["life_expectancy"] = life

    split_years, later_inflation = parse_inflation_scenario(text)
    if split_years is not None and later_inflation is not None:
        overrides["split_after_years"] = split_years
        overrides["later_inflation"] = later_inflation
    else:
        inflation = extract_percent(text, ("通胀率", "通货膨胀率", "物价涨幅"))
        if inflation is not None:
            overrides["inflation"] = inflation

    investment_return = extract_percent(text, ("投资回报率", "投资收益率", "年化收益率", "收益率"))
    if investment_return is not None:
        overrides["investment_return"] = investment_return
    return overrides


def _is_max_return_text(text: str) -> bool:
    return any(k in text for k in ("收益最大", "收益最大化", "收益最高", "最高收益", "追求投资收益", "尽量高收益"))


def _is_low_vol_text(text: str) -> bool:
    return any(k in text for k in ("最小化风险", "风险波动", "波动最小", "最低风险", "稳健配置", "尽量稳健", "降低波动"))


def parse_preference_overrides(text: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if any(k in text for k in ("消费水平不下降", "生活水平不下降", "购买力不下降", "维持当前消费", "保持当前消费")):
        overrides["retirement_goal"] = "maintain_consumption"
    if _is_max_return_text(text):
        overrides["allocation_mode"] = "max_return"
    elif _is_low_vol_text(text):
        overrides["allocation_mode"] = "low_vol"
    if any(k in text for k in ("流动性", "安全性", "稳健", "保守")):
        overrides["risk_preference"] = "low_risk"
    elif any(k in text for k in ("愿意承担风险", "接受波动", "高风险", "进取")):
        overrides["risk_preference"] = "aggressive"
    for product in PRODUCT_NAMES:
        if product in text or fmt_product_name(product) in text:
            if any(k in text for k in ("偏好", "喜欢", "倾向", "关注", "看好", "想配", "配置")):
                overrides["product_preference"] = product
                break
    return overrides


def current_overrides(text: str) -> Dict[str, Any]:
    overrides = parse_scenario_overrides(text)
    overrides.update(parse_preference_overrides(text))
    return overrides


def scenario_context(user_id: str, text: str) -> Dict[str, Any]:
    current = current_overrides(text)
    with _session_lock:
        base = dict(_session_memory.get(user_id, {}))
        if current and has_opinion(text) and not has_assumption(text):
            _session_memory.setdefault(user_id, {}).update(current)
            base.update(current)
    base.update(current)
    debug_log("context", user_id=user_id, current=current, merged=base, remembered=has_opinion(text) and not has_assumption(text))
    return base


INTENT_SUMMARY = {
    "field": "基本信息",
    "behavior_top": "产品偏好",
    "buy_predict": "未来购买倾向",
    "longevity": "长寿风险",
    "retire_in": "退休时间",
    "future_expend": "退休时月支出",
    "living_need": "退休后总需求",
    "min_reserve": "养老资金缺口",
    "pension_pv": "养老金现值",
    "future_acc": "退休时可积累资产",
    "feasibility": "养老目标可行性",
    "max_return": "收益最大化配置",
    "low_vol": "最小化风险波动配置",
}


def _remember_discussed_intent(user_id: str, intent: str) -> None:
    if intent == "report":
        return
    label = INTENT_SUMMARY.get(intent)
    if not label:
        return
    with _session_lock:
        memory = _session_memory.setdefault(user_id, {})
        items = list(memory.get("last_discussed_intents") or [])
        if label in items:
            items.remove(label)
        items.append(label)
        memory["last_discussed_intents"] = items[-6:]


def ctx_life(ctx: Dict[str, Any]) -> int:
    return int(ctx.get("life_expectancy", DEFAULT_LIFE_EXPECTANCY))


def ctx_inflation(ctx: Dict[str, Any]) -> float:
    return float(ctx.get("inflation", DEFAULT_INFLATION))


def ctx_return(ctx: Dict[str, Any]) -> float:
    return float(ctx.get("investment_return", DEFAULT_RETURN))


# ============================================================
# 产品库与配置建议
# ============================================================
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


def count_age_by_text(text: str) -> Optional[int]:
    m = re.search(r"年龄[^0-9]*(?:大于等于|不低于|不少于|至少|及以上|以上|>=)\s*(\d+)", text)
    if m:
        return count_age_ge(int(m.group(1)))
    m = re.search(r"(\d+)\s*岁?[^，。；、]*(?:以上|及以上|大于等于|不低于|不少于)", text)
    if m:
        return count_age_ge(int(m.group(1)))
    m = re.search(r"年龄[^0-9]*(?:大于|超过|>)\s*(\d+)", text)
    if m:
        sql = f"SELECT COUNT(*) AS cnt FROM {BASE_SQL_TABLE} WHERE Age > %s"
        row = query_one(sql, (int(m.group(1)),))
        return int(row.get("cnt") or 0) if row else 0
    m = re.search(r"年龄[^0-9]*(?:小于等于|不超过|至多|<=)\s*(\d+)", text)
    if m:
        sql = f"SELECT COUNT(*) AS cnt FROM {BASE_SQL_TABLE} WHERE Age <= %s"
        row = query_one(sql, (int(m.group(1)),))
        return int(row.get("cnt") or 0) if row else 0
    m = re.search(r"年龄[^0-9]*(?:小于|低于|<)\s*(\d+)", text)
    if m:
        sql = f"SELECT COUNT(*) AS cnt FROM {BASE_SQL_TABLE} WHERE Age < %s"
        row = query_one(sql, (int(m.group(1)),))
        return int(row.get("cnt") or 0) if row else 0
    return None


def parse_life_expectancy(text: str) -> int:
    return parse_life_expectancy_value(text) or DEFAULT_LIFE_EXPECTANCY


def parse_inflation_scenario(text: str) -> Tuple[Optional[int], Optional[float]]:
    m = re.search(r"(\d+)\s*年后.*?(?:通胀率|通货膨胀率|物价涨幅).*?(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2)) / 100


def min_risk_allocation(user: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> str:
    ctx = ctx or {}
    need = retirement_need(
        user,
        ctx_life(ctx),
        ctx_inflation(ctx),
        ctx_return(ctx),
        ctx.get("split_after_years"),
        ctx.get("later_inflation"),
    )
    gap = rnd(need["gap"])
    candidates = []
    for product in allowed_products(str(user.get("Rsk_Cd") or "R1")):
        if product == "年金险":
            continue
        acc = future_accumulation(user, PRODUCTS[product]["return"])
        if acc >= gap:
            # 在满足缺口的合规产品中，取收益率最低的一档（与题面“最低收益率可达标”一致）
            candidates.append((PRODUCTS[product]["return"], PRODUCTS[product]["risk"], product, acc))
    if not candidates:
        return "当前风险承受范围内仅靠现有结余较难覆盖养老缺口，建议提高每月结余或适当提升风险承受能力"
    candidates.sort(key=lambda x: (x[0], x[1]))
    product = candidates[0][2]
    return f"不能，需要改为投资{fmt_product_name(product)}"


def low_vol_allocation(user: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> str:
    ctx = ctx or {}
    need = retirement_need(
        user,
        ctx_life(ctx),
        ctx_inflation(ctx),
        ctx_return(ctx),
        ctx.get("split_after_years"),
        ctx.get("later_inflation"),
    )
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
        return f"{fmt_product_name(product)}配置 100%"
    candidates.sort(key=lambda x: (x[0], x[1]))
    product, acc = candidates[0][2], candidates[0][3]
    main_pct = min(100, pct(gap / acc))
    remaining = 100 - main_pct
    cash = min(10, remaining)
    annuity = remaining - cash
    parts = [f"{fmt_product_name(product)}配置 {main_pct}%"]
    if cash:
        parts.append(f"现金理财 {cash}%")
    if annuity:
        parts.append(f"年金险 {annuity}%")
    return "；".join(parts)


def max_return_allocation(user: Dict[str, Any]) -> str:
    return f"{fmt_product_name(highest_return_product(user))}配置 100%"


def max_return_allocation_detail(user: Dict[str, Any]) -> Dict[str, Any]:
    product = highest_return_product(user)
    acc = future_accumulation(user, PRODUCTS[product]["return"])
    return {
        "main_product": product,
        "main_pct": 100,
        "main_return": PRODUCTS[product]["return"],
        "main_acc": rnd(acc),
        "cash_pct": 0,
        "annuity_pct": 0,
        "mode": "max_return",
    }


def low_vol_allocation_detail(user: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = ctx or {}
    need = retirement_need(
        user,
        ctx_life(ctx),
        ctx_inflation(ctx),
        ctx_return(ctx),
        ctx.get("split_after_years"),
        ctx.get("later_inflation"),
    )
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
        acc = future_accumulation(user, PRODUCTS[product]["return"])
        return {
            "main_product": product,
            "main_pct": 100,
            "main_return": PRODUCTS[product]["return"],
            "main_acc": rnd(acc),
            "cash_pct": 0,
            "annuity_pct": 0,
            "mode": "low_vol",
        }
    candidates.sort(key=lambda x: (x[0], x[1]))
    product, acc = candidates[0][2], candidates[0][3]
    main_pct = min(100, pct(gap / acc))
    remaining = 100 - main_pct
    cash_pct = min(10, remaining)
    annuity_pct = remaining - cash_pct
    return {
        "main_product": product,
        "main_pct": main_pct,
        "main_return": PRODUCTS[product]["return"],
        "main_acc": rnd(acc * main_pct / 100),
        "cash_pct": cash_pct,
        "annuity_pct": annuity_pct,
        "mode": "low_vol",
    }


# ============================================================
# 直接字段问答
# ============================================================
def direct_field_answer(text: str, user: Dict[str, Any]) -> Optional[str]:
    if "年龄" in text or "多大" in text:
        return f"{rnd(money(user.get('Age')))}岁"
    if "性别" in text:
        return str(user.get("Gender") or "")
    if "风险" in text and ("评级" in text or "等级" in text):
        return str(user.get("Rsk_Cd") or "")
    if "净资产" in text:
        return f"{fmt_int(money(user.get('Net_Asset')))}元"
    if "月收入" in text or "每月收入" in text or "收入多少" in text:
        return f"{fmt_int(money(user.get('Monthly_Income')))}元"
    if ("月支出" in text or "每月支出" in text or "开销" in text) and "退休" not in text:
        return f"{fmt_int(money(user.get('Monthly_Expend')))}元"
    if "结余" in text:
        return f"{fmt_int(monthly_surplus(user))}元"
    if "企业年金" in text:
        value = money(user.get("Enterprise_Ann"))
        return "无" if value <= 0 else f"{fmt_int(value)}元"
    if ("退休金" in text or "养老金" in text) and not any(
        word in text for word in (
            "缺口", "最低", "现值", "需要", "规划", "建议书", "积攒", "积累",
            "配置", "测算", "计算", "达成", "支撑", "覆盖", "退休时",
        )
    ):
        return f"{fmt_int(money(user.get('Pension')))}元"
    return None


# ============================================================
# 建议书（数字本地算，文字可选用 LLM 润色）
# ============================================================
def _build_report_facts(
    user_id: str, user: Dict[str, Any], text: str, ctx: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    ctx = ctx or {}
    months = retirement_months(user)
    retire_age_y, retire_age_m = years_months(retirement_age_months(user))
    life = ctx_life(ctx)
    need = retirement_need(
        user,
        life,
        ctx_inflation(ctx),
        ctx_return(ctx),
        ctx.get("split_after_years"),
        ctx.get("later_inflation"),
    )
    pref_product, pref_cnt = top_behavior_product(user_id)
    stated_product_preference = bool(ctx.get("product_preference"))
    if ctx.get("product_preference"):
        pref_product = str(ctx["product_preference"])
        pref_cnt = 0
    allocation_mode = str(ctx.get("allocation_mode") or "low_vol")
    if allocation_mode == "max_return":
        alloc = max_return_allocation(user)
        alloc_detail = max_return_allocation_detail(user)
    else:
        allocation_mode = "low_vol"
        alloc = low_vol_allocation(user, ctx)
        alloc_detail = low_vol_allocation_detail(user, ctx)
    surplus = monthly_surplus(user)
    enterprise = money(user.get("Enterprise_Ann"))
    return {
        "user_id": user_id,
        "age": rnd(money(user.get("Age"))),
        "gender": str(user.get("Gender") or ""),
        "rsk_cd": str(user.get("Rsk_Cd") or ""),
        "net_asset": rnd(money(user.get("Net_Asset"))),
        "monthly_income": rnd(money(user.get("Monthly_Income"))),
        "monthly_expend": rnd(money(user.get("Monthly_Expend"))),
        "surplus": rnd(surplus),
        "pension": rnd(money(user.get("Pension"))),
        "enterprise_ann": rnd(enterprise),
        "life_expectancy": life,
        "inflation": ctx_inflation(ctx),
        "investment_return": ctx_return(ctx),
        "retire_age_y": retire_age_y,
        "retire_age_m": retire_age_m,
        "years_to_retire": years_months_text(months),
        "monthly_exp_at_retire": rnd(need["monthly_exp"]),
        "total_need": rnd(need["living_need"]),
        "pension_pv": rnd(need["pension_pv"]),
        "gap": rnd(need["gap"]),
        "pref_product": pref_product,
        "pref_count": pref_cnt,
        "allocation": alloc,
        "allocation_detail": alloc_detail,
        "allocation_mode": allocation_mode,
        "retirement_goal": ctx.get("retirement_goal", "maintain_consumption"),
        "risk_preference": ctx.get("risk_preference", ""),
        "stated_product_preference": stated_product_preference,
        "last_discussed_intents": list(ctx.get("last_discussed_intents") or []),
    }


def _report_template(f: Dict[str, Any]) -> str:
    enterprise_text = "无" if f["enterprise_ann"] <= 0 else fmt_money(f["enterprise_ann"])
    inflation_text = fmt_rate(f["inflation"])
    detail = f["allocation_detail"]
    pref_product = fmt_product_name(f["pref_product"])
    main_product = fmt_product_name(detail["main_product"])
    years_to_retire = fmt_duration(f["years_to_retire"])
    discussed = "、".join(f.get("last_discussed_intents") or [])
    pref_source = (
        "根据前序沟通中客户表达的产品偏好"
        if f.get("stated_product_preference")
        else (
            f"根据客户的浏览与购买记录（共 {f['pref_count']} 次{pref_product}相关行为）"
            if f["pref_count"]
            else "根据前序沟通信息"
        )
    )
    if f.get("risk_preference") == "aggressive" or f["allocation_mode"] == "max_return":
        preference_desc = "同时客户更关注长期投资收益，可在风险评级允许范围内提升收益弹性。"
    elif f.get("risk_preference") == "low_risk" or f["allocation_mode"] == "low_vol":
        preference_desc = "同时客户更关注稳健、流动性与风险波动控制。"
    else:
        preference_desc = "注重流动性与安全性。"
    discussion_advice_line = (
        f"   - 前序沟通已覆盖{discussed}等议题，建议后续围绕客户目标持续复盘；\n"
        if discussed
        else ""
    )
    if f["allocation_mode"] == "max_return":
        allocation_title = "客户偏好收益最大化方案："
        allocation_lines = (
            f"\n   - 将 100% 配置于{main_product}（年化 {fmt_rate(detail['main_return'])}），"
            f"这是客户当前风险评级 {f['rsk_cd']} 范围内预期收益最高的产品，"
            f"退休时预计可积累约 {fmt_money(detail['main_acc'])}。"
        )
    else:
        allocation_title = "客户偏好最小化风险方案："
        allocation_lines = (
            f"\n   - 将 {detail['main_pct']}% 配置于{main_product}（年化 {fmt_rate(detail['main_return'])}），"
            f"退休时可积累约 {fmt_money(detail['main_acc'])}，能够覆盖 {fmt_money(f['gap'])}养老金缺口；"
        )
    cash_line = ""
    annuity_line = ""
    if f["allocation_mode"] != "max_return" and detail["cash_pct"]:
        cash_line = (
            f"\n   - 将 {detail['cash_pct']}% 配置于现金理财（年化 {fmt_rate(PRODUCTS['现金理财']['return'])}），"
            "用于应对意外事件的流动性储备，并符合客户的投资偏好；"
        )
    if f["allocation_mode"] != "max_return" and detail["annuity_pct"]:
        annuity_line = (
            f"\n   - 将 {detail['annuity_pct']}% 配置于年金险产品（IRR {fmt_rate(PRODUCTS['年金险']['return'])}），"
            "用于对冲长寿风险，退休后可领取终身年金。"
        )
    return (
        "1. 基本情况\n"
        f"客户 ID：{f['user_id']}，年龄：{f['age']} 岁，性别：{f['gender']}，"
        f"风险评级：{f['rsk_cd']}。当前净资产：{fmt_money(f['net_asset'])}，"
        f"每月结余：{fmt_money(f['surplus'])}（月收入 {fmt_money(f['monthly_income'])} - "
        f"月支出 {fmt_money(f['monthly_expend'])}）。每月退休金：{fmt_money(f['pension'])}，"
        f"企业年金（一次性提取）：{enterprise_text}。\n"
        "2. 基本假设\n"
        f"预期寿命 {f['life_expectancy']} 岁，长期通胀率 {inflation_text}，"
        f"退休年龄 {f['retire_age_y']} 岁"
        f"{str(f['retire_age_m']) + ' 个月' if f['retire_age_m'] else ''}"
        f"（距退休 {years_to_retire}）。\n"
        "3. 养老目标\n"
        f"每月可花费与当前 {fmt_money(f['monthly_expend'])}购买力相同的金额"
        f"（退休时约为 {fmt_money(f['monthly_exp_at_retire'])}）。\n"
        "4. 退休后财富需求测算\n"
        f"退休后预计总需求 {fmt_money(f['total_need'])}，其中养老金（先付年金现值）"
        f"可支撑 {fmt_money(f['pension_pv'])}，还有 {fmt_money(f['gap'])}缺口需要通过投资积累来覆盖。\n"
        "5. 产品偏好\n"
        f"{pref_source}，推测客户偏好{pref_product}类产品，{preference_desc}\n"
        "6. 资产配置方式与具体方案\n"
        f"{allocation_title}"
        f"{allocation_lines}"
        f"{cash_line}{annuity_line}\n"
        "7. 其他建议\n"
        f"{discussion_advice_line}"
        f"   - 客户目前 {f['age']} 岁，距退休长达 {years_to_retire}，复利效应显著，建议尽早开始投资积累；\n"
        f"   - 在沟通中了解客户对{pref_product}偏好的原因，视情况适当增减{pref_product}的配置比例；\n"
        "   - 随着客户投资能力与经验积累，若风险评级提升，还可增加权益类产品的配置以获取更高的长期收益。"
    )


def _llm_polish_report(facts: Dict[str, Any], baseline: str) -> Optional[str]:
    """让大模型在保留全部数字的前提下润色建议书。
    若模型未配置、超时、或丢失关键数字，则返回 None 由模板兜底。"""
    if not LLM_ENABLED or not LLM_REPORT_POLISH:
        debug_log(
            "llm_call_skipped",
            call_site="_llm_polish_report",
            purpose="在模板建议书基础上润色养老规划建议书",
            reason="LLM_ENABLED is false or ONE_API_REPORT_POLISH is false",
            user_id=facts.get("user_id"),
        )
        return None
    system = (
        "你是资深财富顾问。请根据给定客户事实，撰写一份养老规划建议书。"
        "严格要求：\n"
        "1. 必须包含 7 个章节，章节标题与示例一致：1. 基本情况；2. 基本假设；"
        "3. 养老目标；4. 退休后财富需求测算；5. 产品偏好；6. 资产配置方式与具体方案；7. 其他建议。\n"
        "2. 一切金额/年龄/比例数字必须与给定事实完全一致，不允许改动、四舍五入、合并或省略。\n"
        "3. 输出为纯文本，不要 Markdown 标题符号、不要表格、不要解释、不要多余前后缀。\n"
        "4. 篇幅控制在 500 字以内。"
    )
    user_prompt = (
        f"客户事实（JSON）：\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"参考底稿（数字以此为准）：\n{baseline}\n\n"
        "请基于事实和底稿，输出最终建议书。"
    )
    text = llm_chat(
        user_prompt,
        system_prompt=system,
        max_tokens=900,
        timeout=15,
        purpose="在模板建议书基础上润色养老规划建议书",
        call_site="_llm_polish_report",
    )
    if not text:
        debug_log(
            "llm_result_rejected",
            call_site="_llm_polish_report",
            purpose="在模板建议书基础上润色养老规划建议书",
            reason="empty response",
            user_id=facts.get("user_id"),
        )
        return None
    required_numbers = [
        str(facts["age"]), str(facts["net_asset"]), str(facts["monthly_income"]),
        str(facts["monthly_expend"]), str(facts["surplus"]), str(facts["pension"]),
        str(facts["monthly_exp_at_retire"]), str(facts["total_need"]),
        str(facts["pension_pv"]), str(facts["gap"]),
    ]
    if not all(num in text for num in required_numbers):
        debug_log(
            "llm_result_rejected",
            call_site="_llm_polish_report",
            purpose="在模板建议书基础上润色养老规划建议书",
            reason="missing required numbers",
            user_id=facts.get("user_id"),
            required_numbers=required_numbers,
        )
        return None
    if facts["allocation"].split("；")[0] not in text:
        debug_log(
            "llm_result_rejected",
            call_site="_llm_polish_report",
            purpose="在模板建议书基础上润色养老规划建议书",
            reason="missing allocation summary",
            user_id=facts.get("user_id"),
            allocation=facts["allocation"],
        )
        return None
    debug_log(
        "llm_result_used",
        call_site="_llm_polish_report",
        purpose="在模板建议书基础上润色养老规划建议书",
        user_id=facts.get("user_id"),
    )
    return text


def recommendation_report(
    user_id: str, user: Dict[str, Any], text: str, ctx: Optional[Dict[str, Any]] = None
) -> str:
    facts = _build_report_facts(user_id, user, text, ctx)
    baseline = _report_template(facts)
    polished = _llm_polish_report(facts, baseline)
    return polished or baseline


# ============================================================
# 开放式建议（短答）：可调用 LLM，并要求只返回一句话
# ============================================================
def llm_short_advice(text: str, user: Optional[Dict[str, Any]]) -> Optional[str]:
    if not LLM_ENABLED or not LLM_FALLBACK:
        debug_log(
            "llm_call_skipped",
            call_site="llm_short_advice",
            purpose="开放式养老规划问题短答兜底",
            reason="LLM_ENABLED is false or ONE_API_FALLBACK is false",
            question=text,
        )
        return None
    profile = ""
    if user:
        profile = (
            f"客户信息：年龄{rnd(money(user.get('Age')))}岁，性别{user.get('Gender')}，"
            f"风险评级{user.get('Rsk_Cd')}，月收入{fmt_int(money(user.get('Monthly_Income')))}元，"
            f"月支出{fmt_int(money(user.get('Monthly_Expend')))}元，"
            f"净资产{fmt_int(money(user.get('Net_Asset')))}元，"
            f"退休金{fmt_int(money(user.get('Pension')))}元。\n"
        )
    system = (
        "你是养老规划Agent。请根据养老规划逻辑和产品库（现金理财1.5%、定期存款2.0%、"
        "短债类产品2.4%、固收+产品4.25%、权益类产品6.0%、年金险2.5%）回答客户经理的问题。"
        "要求：只输出最终答案，不超过 30 字，不要解释、不要前后缀、不要标点列表。"
    )
    prompt = profile + f"问题：{text}\n答："
    res = llm_chat(
        prompt,
        system_prompt=system,
        max_tokens=120,
        timeout=12,
        purpose="开放式养老规划问题短答兜底",
        call_site="llm_short_advice",
    )
    if not res:
        debug_log(
            "llm_result_rejected",
            call_site="llm_short_advice",
            purpose="开放式养老规划问题短答兜底",
            reason="empty response",
            question=text,
        )
        return None
    answer = res.splitlines()[0].strip()
    debug_log(
        "llm_result_used",
        call_site="llm_short_advice",
        purpose="开放式养老规划问题短答兜底",
        question=text,
        answer=answer,
    )
    return answer


# ============================================================
# 单客户问题路由
# ============================================================
def _is_report_question(text: str) -> bool:
    if any(k in text for k in ("建议书", "规划书", "方案书")):
        return True
    if "养老规划" in text and any(k in text for k in ("生成", "出具", "做", "撰写", "给出")):
        return True
    return False


def _is_feasibility_question(text: str) -> bool:
    keywords_yes = ("能否", "能不能", "能达成", "是否能", "可否", "是否可以", "能实现", "能完成")
    if any(k in text for k in keywords_yes):
        return True
    return "不能" in text and "调整" in text


def _is_retire_in_question(text: str) -> bool:
    if "距离退休" in text:
        return True
    return "退休" in text and (
        "还有多久" in text or "还要多久" in text or "什么时候" in text
        or "退休年龄" in text or "多久后" in text or "几年后" in text
    )


def _is_future_expend_question(text: str) -> bool:
    if any(k in text for k in ("刚退休", "退休当月", "退休那个月", "退休的时候")) and not any(
        k in text for k in ("最低", "至少", "缺口", "积攒", "储备", "准备")
    ):
        return True
    if ("退休时" in text or "退休后" in text) and ("最低" not in text):
        return any(k in text for k in (
            "每月", "月支出", "月花销", "月花费", "月消费", "月开销",
            "支出多少", "花多少钱", "消费多少", "每个月",
        ))
    return False


def _is_living_need_question(text: str) -> bool:
    if any(k in text for k in ("最低", "至少", "缺口", "还差", "养老金现值", "退休金现值")):
        return False
    return (
        "退休后" in text
        and any(k in text for k in (
            "总需求", "总支出", "总花销", "总花费", "总消费", "生活费",
            "财富需求", "资金需求", "一共要花", "总开支", "总费用",
        ))
    )


def _is_min_reserve_question(text: str) -> bool:
    if any(k in text for k in ("哪些", "什么问题", "沟通重点", "关注点")) and not any(
        k in text for k in ("多少钱", "多少元", "金额", "缺口", "差额", "还差")
    ):
        return False
    if "现值" in text and ("养老金" in text or "退休金" in text):
        return False
    if any(k in text for k in ("缺口", "还差", "差额", "资金缺口", "养老缺口", "差多少钱", "还需要多少钱")):
        return True
    if any(k in text for k in ("本金", "本钱", "养老资金", "养老储备", "养老本金", "养老准备金")):
        return True
    return (
        any(k in text for k in ("最低", "至少", "最少", "需要", "应该", "要"))
        and any(k in text for k in ("积攒", "攒", "准备", "储备", "存", "留"))
        and not any(k in text for k in ("每月", "月支出", "月花销", "月花费"))
    )


def _is_accumulate_question(text: str) -> bool:
    return any(k in text for k in (
        "可以积攒", "能积攒", "积攒下", "可积累", "能积累", "可以积累",
        "退休时资产", "退休时能有", "退休时有多少钱", "退休时能攒", "退休时能存",
        "预计积累", "预计攒下", "可以攒下", "能够积累", "退休时手里有",
        "退休时账户有", "退休时资产有", "退休时可攒", "退休时可存",
    ))


def _is_top_behavior_question(text: str) -> bool:
    if any(k in text for k in ("行为最多", "最多行为", "最常看", "最常买", "最关注")):
        return True
    return any(k in text for k in ("偏好", "喜欢", "倾向")) and "产品" in text


def _is_buy_predict_question(text: str) -> bool:
    if any(k in text for k in ("可能购买", "购买预测", "预测购买", "最可能买", "可能买", "会买什么")):
        return True
    return ("未来" in text or "接下来" in text or "一个星期" in text or "下周" in text or "下个月" in text) and (
        "购买" in text or "买" in text
    )


def _is_longevity_question(text: str) -> bool:
    return "寿命" in text and any(k in text for k in ("增加", "配置", "延长", "对冲"))


def _is_max_return_question(text: str) -> bool:
    return _is_max_return_text(text)


def _is_low_vol_question(text: str) -> bool:
    return _is_low_vol_text(text)


def _is_pension_pv_question(text: str) -> bool:
    return any(k in text for k in ("养老金", "退休金", "社保")) and any(k in text for k in ("现值", "折现", "可支撑", "能支撑", "覆盖多少"))


def _is_context_update_only(text: str, ctx: Dict[str, Any]) -> bool:
    if not ctx or not has_opinion(text) or has_assumption(text):
        return False
    question_words = (
        "多少", "几", "吗", "能否", "是否", "怎么", "如何", "建议", "建议书",
        "规划", "配置", "测算", "计算", "出具", "生成", "需要", "缺口",
    )
    return not any(word in text for word in question_words)


def _route_intent(text: str) -> Optional[str]:
    if _is_report_question(text):
        return "report"
    if _is_top_behavior_question(text):
        return "behavior_top"
    if _is_buy_predict_question(text):
        return "buy_predict"
    if _is_longevity_question(text):
        return "longevity"
    if _is_retire_in_question(text):
        return "retire_in"
    if _is_future_expend_question(text):
        return "future_expend"
    if _is_living_need_question(text):
        return "living_need"
    if _is_pension_pv_question(text):
        return "pension_pv"
    if _is_feasibility_question(text):
        return "feasibility"
    if _is_min_reserve_question(text):
        return "min_reserve"
    if _is_accumulate_question(text):
        return "future_acc"
    if _is_max_return_question(text):
        return "max_return"
    if _is_low_vol_question(text):
        return "low_vol"
    return None


def _answer_by_intent(
    intent: str, text: str, user_id: str, user: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    ctx = ctx or {}
    if intent == "report":
        return recommendation_report(user_id, user, text, ctx)
    if intent == "behavior_top":
        return top_behavior_product(user_id)[0]
    if intent == "buy_predict":
        return top_behavior_product(user_id, buy_first=True)[0]
    if intent == "longevity":
        return "年金险"
    if intent == "retire_in":
        return years_months_text(retirement_months(user))
    if intent == "future_expend":
        return fmt_short_money(
            future_monthly_expend(user, ctx_inflation(ctx), ctx.get("split_after_years"), ctx.get("later_inflation"))
        )
    if intent == "living_need":
        need = retirement_need(
            user,
            ctx_life(ctx),
            ctx_inflation(ctx),
            ctx_return(ctx),
            ctx.get("split_after_years"),
            ctx.get("later_inflation"),
        )
        return fmt_short_money(need["living_need"])
    if intent == "min_reserve":
        need = retirement_need(
            user,
            ctx_life(ctx),
            ctx_inflation(ctx),
            ctx_return(ctx),
            ctx.get("split_after_years"),
            ctx.get("later_inflation"),
        )
        return fmt_short_money(need["gap"])
    if intent == "pension_pv":
        need = retirement_need(
            user,
            ctx_life(ctx),
            ctx_inflation(ctx),
            ctx_return(ctx),
            ctx.get("split_after_years"),
            ctx.get("later_inflation"),
        )
        return fmt_short_money(need["pension_pv"])
    if intent == "future_acc":
        return fmt_short_money(future_accumulation(user, ctx_return(ctx)))
    if intent == "feasibility":
        need = rnd(
            retirement_need(
                user,
                ctx_life(ctx),
                ctx_inflation(ctx),
                ctx_return(ctx),
                ctx.get("split_after_years"),
                ctx.get("later_inflation"),
            )["gap"]
        )
        acc = rnd(future_accumulation(user, ctx_return(ctx)))
        if acc >= need:
            return "能达成目标"
        return min_risk_allocation(user, ctx)
    if intent == "max_return":
        return max_return_allocation(user)
    if intent == "low_vol":
        return low_vol_allocation(user, ctx)
    return None


def answer_user_question(text: str, user_id: str, user: Dict[str, Any]) -> str:
    ctx = scenario_context(user_id, text)
    if _is_context_update_only(text, current_overrides(text)):
        debug_log("answer", user_id=user_id, intent="context_update", answer="已记录客户观点")
        return "已记录客户观点"
    direct = direct_field_answer(text, user)
    if direct is not None:
        _remember_discussed_intent(user_id, "field")
        debug_log("answer", user_id=user_id, intent="field", answer=direct)
        return direct

    intent = _route_intent(text)
    if intent:
        ans = _answer_by_intent(intent, text, user_id, user, ctx)
        if ans is not None:
            _remember_discussed_intent(user_id, intent)
            debug_log("answer", user_id=user_id, intent=intent, answer=ans)
            return ans

    # 规则没匹配上，尝试让大模型理解自然语言变体；大模型只做意图分类，答案仍由代码确定性计算。
    llm_intent = llm_classify_intent(text)
    if llm_intent:
        ans = _answer_by_intent(llm_intent, text, user_id, user, ctx)
        if ans is not None:
            _remember_discussed_intent(user_id, llm_intent)
            debug_log("answer", user_id=user_id, intent=f"llm:{llm_intent}", answer=ans)
            return ans

    # 最后兜底：不让大模型自由短答，直接生成结构化建议书
    ans = recommendation_report(user_id, user, text, ctx)
    debug_log("answer", user_id=user_id, intent="fallback_report", answer=ans[:200])
    return ans


# ============================================================
# 全表统计类问题（无客户ID）
# ============================================================
def _try_population_question(text: str) -> Optional[str]:
    if (
        "年龄" in text
        and ("多少客户" in text or "多少个客户" in text or "客户年龄" in text or "客户数" in text or "人数" in text)
    ):
        cnt = count_age_by_text(text)
        if cnt is not None:
            return f"{cnt}个"
    if "权益类产品" in text and "平均年龄" in text:
        min_count = 2
        m = re.search(r"(\d+)\s*次", text)
        if m:
            min_count = int(m.group(1))
        return f"{avg_age_for_equity_views(min_count)}岁"
    return None


def _remember_last_user(user_id: str) -> None:
    global _last_user_id
    with _session_lock:
        _last_user_id = user_id


def _infer_last_user(text: str) -> Optional[str]:
    if not any(k in text for k in ("该客户", "这位客户", "这个客户", "他", "她", "上述", "上面", "前面")):
        return None
    with _session_lock:
        return _last_user_id


# ============================================================
# 入口
# ============================================================
def run(inf: str) -> str:
    text = (inf or "").strip()
    if not text:
        return ""
    try:
        debug_log("question", question=text)
        pop = _try_population_question(text)
        if pop is not None:
            debug_log("answer", user_id=None, intent="population", answer=pop)
            return pop

        user_id = extract_user_id(text) or _infer_last_user(text)
        if not user_id:
            debug_log("answer", user_id=None, intent="missing_user", question=text)
            return "请提供客户ID"
        user = get_user(user_id)
        if not user:
            debug_log("answer", user_id=user_id, intent="missing_user_record", question=text)
            return "未查询到客户信息"
        _remember_last_user(user_id)
        return answer_user_question(text, user_id, user)
    except Exception as exc:
        debug_log("error", question=text, error=repr(exc))
        return "暂时无法回答该问题"
    finally:
        _close_conn()


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.stdout.write(run(question))


if __name__ == "__main__":
    main()
