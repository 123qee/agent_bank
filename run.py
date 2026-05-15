import csv
import glob
import math
import os
import re
import sqlite3
from pathlib import Path


CURRENT_YEAR = 2025
CURRENT_MONTH = 3
CURRENT_DAY = 31

DEFAULT_INFLATION = 0.02
DEFAULT_RETURN = 0.02
DEFAULT_LIFE_EXPECTANCY = 80

SESSION = {
    "users": {}
}

PRODUCTS = {
    "现金理财": {"rate": 0.015, "risk": 1},
    "定期存款": {"rate": 0.020, "risk": 1},
    "短债类产品": {"rate": 0.024, "risk": 2},
    "年金险": {"rate": 0.025, "risk": 3},
    "固收+产品": {"rate": 0.0425, "risk": 3},
    "权益类产品": {"rate": 0.060, "risk": 5},
}

RISK_CAP = {
    "R1": 1, "1": 1,
    "R2": 2, "2": 2,
    "R3": 3, "3": 3,
    "R4": 4, "4": 4,
    "R5": 5, "5": 5,
}


def _norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(s).lower())


def _money(x):
    return f"{int(round(float(x)))} 元"


def _percent(x):
    return f"{int(math.ceil(float(x) * 100 - 1e-12))}%"


class DataStore:
    def __init__(self):
        self.conn = None
        self.base_table = None
        self.action_table = None
        self.base_cols = []
        self.action_cols = []
        self.base_map = {}
        self.action_map = {}
        self._connect()

    def _connect(self):
        db_paths = []
        for root in ["/work/data/task2", "/work/task2", "/work/data", os.getcwd(), "."]:
            for pat in ("*.db", "*.sqlite", "*.sqlite3"):
                db_paths.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
        for db in db_paths:
            try:
                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                base = self._pick_table(tables, ["base_table", "base", "customer", "客户"])
                action = self._pick_table(tables, ["action_table", "action", "behavior", "行为"])
                if base and action:
                    self.conn = conn
                    self.base_table = base
                    self.action_table = action
                    self._init_columns()
                    return
                conn.close()
            except Exception:
                pass
        self._load_csv_to_memory()

    def _load_csv_to_memory(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        csv_paths = []
        for root in ["/work/data/task2", "/work/task2", "/work/data", os.getcwd(), "."]:
            csv_paths.extend(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        base_csv = self._pick_path(csv_paths, ["base", "customer", "客户"])
        action_csv = self._pick_path(csv_paths, ["action", "behavior", "train", "行为"])
        if base_csv:
            self._import_csv(conn, base_csv, "base_table")
            self.base_table = "base_table"
        if action_csv:
            self._import_csv(conn, action_csv, "action_table")
            self.action_table = "action_table"
        self.conn = conn
        self._init_columns()

    @staticmethod
    def _pick_table(tables, keys):
        lower = {t.lower(): t for t in tables}
        for key in keys:
            if key in lower:
                return lower[key]
        for t in tables:
            nt = _norm(t)
            if any(_norm(k) in nt for k in keys):
                return t
        return None

    @staticmethod
    def _pick_path(paths, keys):
        for p in paths:
            name = _norm(os.path.basename(p))
            if any(_norm(k) in name for k in keys):
                return p
        return None

    @staticmethod
    def _import_csv(conn, path, table):
        def qid(name):
            return '"' + str(name).replace('"', '""') + '"'

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            raw_cols = reader.fieldnames or []
            seen = {}
            cols = []
            for i, col in enumerate(raw_cols):
                name = col or f"c{i}"
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 0
                cols.append(name)
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(
                f'CREATE TABLE "{table}" ({",".join([qid(c) + " TEXT" for c in cols])})'
            )
            sql = f'INSERT INTO "{table}" ({",".join(qid(c) for c in cols)}) VALUES ({",".join(["?"] * len(cols))})'
            rows = ([row.get(raw_cols[i], "") for i in range(len(cols))] for row in reader)
            conn.executemany(sql, rows)
            conn.commit()

    def _init_columns(self):
        self.base_cols = self._columns(self.base_table)
        self.action_cols = self._columns(self.action_table)
        self.base_map = self._build_map(self.base_cols, {
            "user_id": ["user_id", "userid", "cust_id", "customer_id", "客户id", "用户id"],
            "age": ["age", "年龄"],
            "gender": ["gender", "sex", "性别"],
            "risk": ["risk", "risk_level", "rsk_lvl", "风险", "风险等级", "风险评级"],
            "asset": ["net_asset", "asset", "assets", "净资产", "资产"],
            "income": ["monthly_income", "income", "月收入", "收入"],
            "expense": ["monthly_expense", "expense", "月支出", "支出", "消费"],
            "pension": ["pension", "monthly_pension", "social_pension", "退休金", "养老金", "社保"],
            "annuity": ["enterprise_annuity", "corp_annuity", "annuity", "企业年金"],
        })
        self.action_map = self._build_map(self.action_cols, {
            "user_id": ["user_id", "userid", "cust_id", "customer_id", "客户id", "用户id"],
            "act_typ": ["act_typ", "action_type", "act_type", "行为类型", "动作"],
            "prod_typ": ["prod_typ", "product_type", "prod_type", "产品类型"],
            "prod_sub_typ": ["prod_sub_typ", "product_sub_type", "prod_sub_type", "产品子类型"],
            "rsk_lvl": ["rsk_lvl", "risk_level", "risk", "产品风险", "风险等级"],
        })

    def _columns(self, table):
        if not table:
            return []
        try:
            return [r[1] for r in self.conn.execute(f'PRAGMA table_info("{table}")')]
        except Exception:
            return []

    @staticmethod
    def _build_map(cols, spec):
        normalized = {_norm(c): c for c in cols}
        out = {}
        for key, names in spec.items():
            for name in names:
                nn = _norm(name)
                if nn in normalized:
                    out[key] = normalized[nn]
                    break
            if key not in out:
                for c in cols:
                    nc = _norm(c)
                    if any(_norm(name) in nc for name in names):
                        out[key] = c
                        break
        return out

    def q(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql, params=(), default=0):
        rows = self.q(sql, params)
        if not rows:
            return default
        val = rows[0][0]
        return default if val is None else val

    def col(self, scope, name):
        mp = self.base_map if scope == "base" else self.action_map
        return mp.get(name)

    def user(self, user_id):
        uid = self.col("base", "user_id")
        if not (self.base_table and uid):
            return {}
        rows = self.q(f'SELECT * FROM "{self.base_table}" WHERE "{uid}"=? LIMIT 1', (user_id,))
        return dict(rows[0]) if rows else {}


DB = None


def db():
    global DB
    if DB is None:
        DB = DataStore()
    return DB


def val(row, field, default=0):
    d = db()
    col = d.col("base", field)
    if not col or col not in row or row[col] in (None, ""):
        return default
    v = row[col]
    if field in ("age", "asset", "income", "expense", "pension", "annuity"):
        try:
            return float(str(v).replace(",", ""))
        except Exception:
            return default
    return str(v)


def user_id_from(text):
    m = re.search(r"V\d+", text, re.I)
    return m.group(0).upper() if m else None


def number_before(text, word, default=None):
    m = re.search(r"(\d+(?:\.\d+)?)\s*" + word, text)
    return float(m.group(1)) if m else default


def risk_cap(row):
    risk = val(row, "risk", "R3")
    return RISK_CAP.get(str(risk).upper(), 3)


def gender(row):
    g = val(row, "gender", "")
    if "女" in g or str(g).upper() in ("F", "FEMALE"):
        return "女"
    return "男"


def retirement_delay_months(age, g):
    base_age = 60 if g == "男" else 55
    original_year = CURRENT_YEAR - int(age) + base_age
    original_month = CURRENT_MONTH
    delta = (original_year - 2025) * 12 + (original_month - 1)
    return max(0, min(36, int(math.ceil(delta / 4.0))))


def retirement_info(row):
    age = int(round(val(row, "age", 0)))
    g = gender(row)
    base_age = 60 if g == "男" else 55
    delay = retirement_delay_months(age, g)
    retire_age_months = base_age * 12 + delay
    months_left = max(0, retire_age_months - age * 12)
    return {
        "age": age,
        "gender": g,
        "delay": delay,
        "retire_age_months": retire_age_months,
        "months_left": months_left,
        "retire_years": retire_age_months // 12,
        "retire_month_rem": retire_age_months % 12,
    }


def monthly_rate(annual):
    return annual / 12.0


def future_expense(row, params=None):
    params = params or {}
    expense = val(row, "expense", 0)
    months = retirement_info(row)["months_left"]
    split = params.get("inflation_after_years")
    if split is not None:
        first = min(months, int(split * 12))
        second = max(0, months - first)
        i1 = params.get("inflation_before", DEFAULT_INFLATION)
        i2 = params.get("inflation_after", DEFAULT_INFLATION)
        return expense * (1 + i1 / 12) ** first * (1 + i2 / 12) ** second
    return expense * (1 + DEFAULT_INFLATION / 12) ** months


def pv_annuity(payment, months, discount_monthly):
    if months <= 0:
        return 0.0
    if abs(discount_monthly) < 1e-12:
        return payment * months
    return payment * (1 - (1 + discount_monthly) ** (-months)) / discount_monthly


def living_need(row, params=None):
    params = params or {}
    info = retirement_info(row)
    life = params.get("life_expectancy", DEFAULT_LIFE_EXPECTANCY)
    retire_months = max(0, int(round(life * 12 - info["retire_age_months"])))
    invest = params.get("return_rate", DEFAULT_RETURN)
    post_inflation = params.get("post_inflation", params.get("inflation_after", DEFAULT_INFLATION))
    e = future_expense(row, params)
    if "inflation_after_years" not in params and abs(post_inflation - DEFAULT_INFLATION) < 1e-12:
        e = round(e)
    factor = (1 + post_inflation / 12) / (1 + invest / 12)
    living_pv = sum(e * (factor ** k) for k in range(retire_months))
    pension = val(row, "pension", 0)
    pension_pv = sum(pension / ((1 + post_inflation / 12) ** k) for k in range(retire_months))
    annuity = val(row, "annuity", 0)
    gap = max(0.0, living_pv - pension_pv - annuity)
    return {
        "monthly_expense_at_retire": e,
        "retire_months": retire_months,
        "living_pv": living_pv,
        "pension_pv": pension_pv,
        "gap": gap,
    }


def future_accumulation(row, annual_return=DEFAULT_RETURN):
    info = retirement_info(row)
    months = info["months_left"]
    r = annual_return / 12
    asset = val(row, "asset", 0)
    surplus = val(row, "income", 0) - val(row, "expense", 0)
    if abs(r) < 1e-12:
        return asset + surplus * months
    return asset * (1 + r) ** months + surplus * ((1 + r) ** months - 1) / r


def allowed_products(row):
    cap = risk_cap(row)
    return {k: v for k, v in PRODUCTS.items() if v["risk"] <= cap}


def product_case_sql():
    d = db()
    a = d.action_map
    prod_typ = a.get("prod_typ")
    prod_sub = a.get("prod_sub_typ")
    rsk = a.get("rsk_lvl")
    if not (prod_typ and prod_sub and rsk):
        return "'其他'"
    return f"""CASE
        WHEN "{rsk}" IN ('R4','R5') AND "{prod_typ}"='基金' THEN '权益类产品'
        WHEN "{rsk}"='R2' AND "{prod_typ}" IN ('基金','理财') THEN '短债类产品'
        WHEN "{rsk}"='R3' AND "{prod_typ}" IN ('基金','理财') THEN '固收+产品'
        WHEN "{prod_sub}"='一般性' AND "{prod_typ}"='存款' THEN '定期存款'
        WHEN "{prod_sub}"='现金' THEN '现金理财'
        WHEN "{prod_sub}" IN ('税延养老年金','养老年金') THEN '年金险'
        ELSE '其他' END"""


def top_product(user_id, only_buy=False, only_view=False):
    d = db()
    uid = d.col("action", "user_id")
    act = d.col("action", "act_typ")
    prod_typ = d.col("action", "prod_typ")
    if not (d.action_table and uid):
        return "现金理财", 0
    where = [f'"{uid}"=?']
    params = [user_id]
    if prod_typ:
        where.append(f'("{prod_typ}" IS NULL OR "{prod_typ}"<>"非财富")')
    if act and only_buy:
        where.append(f'"{act}" LIKE "%购买%"')
    if act and only_view:
        where.append(f'("{act}" LIKE "%浏览%" OR "{act}" LIKE "%查看%")')
    case = product_case_sql()
    sql = f'''SELECT {case} AS product, COUNT(*) AS cnt
              FROM "{d.action_table}"
              WHERE {" AND ".join(where)}
              GROUP BY product
              ORDER BY cnt DESC, product
              LIMIT 1'''
    rows = d.q(sql, params)
    if rows and rows[0]["product"] != "其他":
        return rows[0]["product"], int(rows[0]["cnt"])
    if only_buy:
        return top_product(user_id, only_buy=False)
    return "现金理财", 0


def count_age_ge(age):
    d = db()
    col_age = d.col("base", "age")
    if not (d.base_table and col_age):
        return 0
    return int(d.scalar(f'SELECT COUNT(*) FROM "{d.base_table}" WHERE CAST("{col_age}" AS REAL)>=?', (age,), 0))


def avg_age_for_product(product, min_count=1, only_view=True):
    d = db()
    buid = d.col("base", "user_id")
    auid = d.col("action", "user_id")
    age = d.col("base", "age")
    act = d.col("action", "act_typ")
    if not (buid and auid and age):
        return 0
    case = product_case_sql()
    where = [f"{case}=?"]
    params = [product]
    if only_view and act:
        where.append(f'("{act}" LIKE "%浏览%" OR "{act}" LIKE "%查看%")')
    sql = f'''WITH T AS (
                SELECT "{auid}" AS user_id, COUNT(*) AS cnt
                FROM "{d.action_table}"
                WHERE {" AND ".join(where)}
                GROUP BY "{auid}" HAVING cnt>=?
              )
              SELECT AVG(CAST(b."{age}" AS REAL))
              FROM T INNER JOIN "{d.base_table}" b ON b."{buid}"=T.user_id'''
    return float(d.scalar(sql, tuple(params + [min_count]), 0))


def parse_params(text):
    params = {}
    life = re.search(r"(?:寿命|人均寿命|预期寿命).*?(\d{2,3})\s*岁", text)
    if life:
        params["life_expectancy"] = int(life.group(1))
    m = re.search(r"(\d+)\s*年后.*?通胀率.*?(?:提升到|变为|到)\s*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        params["inflation_after_years"] = int(m.group(1))
        params["inflation_before"] = DEFAULT_INFLATION
        params["inflation_after"] = float(m.group(2)) / 100
        params["post_inflation"] = float(m.group(2)) / 100
    else:
        m2 = re.search(r"通胀率.*?(\d+(?:\.\d+)?)\s*%", text)
        if m2:
            params["inflation_after"] = float(m2.group(1)) / 100
            params["post_inflation"] = float(m2.group(1)) / 100
    r = re.search(r"(?:收益率|回报率|投资回报率).*?(\d+(?:\.\d+)?)\s*%", text)
    if r:
        params["return_rate"] = float(r.group(1)) / 100
    return params


def remember(text, user_id):
    if not user_id:
        return
    mem = SESSION["users"].setdefault(user_id, {})
    has_if = any(w in text for w in ("假如", "假设", "如果", "若", "倘若"))
    if "最小化风险" in text or "风险波动" in text:
        mem["allocation_preference"] = "min_risk"
    elif ("收益最大化" in text or "追求投资收益" in text) and not has_if:
        mem["allocation_preference"] = "max_return"
    if ("消费水平不下降" in text or "维持消费" in text) and not has_if:
        mem["goal"] = "维持退休后消费水平不下降"
    if ("预期寿命" in text or "人均寿命" in text) and not has_if:
        life = parse_params(text).get("life_expectancy")
        if life:
            mem["life_expectancy"] = life


def recommend_adjustment(row, params=None):
    need = living_need(row, params)["gap"]
    for name, info in sorted(allowed_products(row).items(), key=lambda kv: (kv[1]["rate"], kv[1]["risk"])):
        fv = future_accumulation(row, info["rate"])
        if fv + 1e-6 >= need:
            return name, fv, need
    name, info = max(allowed_products(row).items(), key=lambda kv: kv[1]["rate"])
    return name, future_accumulation(row, info["rate"]), need


def allocation_min_risk(row, params=None):
    main, fv, need = recommend_adjustment(row, params)
    if need <= 0:
        return "现金理财配置 100%"
    pct = min(100, int(math.ceil(need / max(fv, 1) * 100 - 1e-12)))
    if pct >= 90:
        return f"{main}配置 {pct}%；现金理财 {100 - pct}%"
    return f"{main}配置 {pct}%；现金理财 10%；年金险 {90 - pct}%"


def allocation_max_return(row):
    name = max(allowed_products(row).items(), key=lambda kv: kv[1]["rate"])[0]
    return f"{name}配置 100%"


def basic_field_answer(text, row):
    fields = [
        ("年龄", "age", "岁"),
        ("性别", "gender", ""),
        ("风险", "risk", ""),
        ("风险评级", "risk", ""),
        ("风险等级", "risk", ""),
        ("净资产", "asset", " 元"),
        ("资产", "asset", " 元"),
        ("月收入", "income", " 元"),
        ("收入", "income", " 元"),
        ("月支出", "expense", " 元"),
        ("支出", "expense", " 元"),
        ("退休金", "pension", " 元"),
        ("养老金", "pension", " 元"),
        ("企业年金", "annuity", " 元"),
    ]
    for key, field, unit in fields:
        if key in text:
            v = val(row, field, "")
            if field in ("age", "asset", "income", "expense", "pension", "annuity"):
                return f"{int(round(float(v)))}{unit}"
            return f"{v}{unit}"
    return None


def report(user_id, row):
    mem = SESSION["users"].get(user_id, {})
    params = {}
    if "life_expectancy" in mem:
        params["life_expectancy"] = mem["life_expectancy"]
    info = retirement_info(row)
    need = living_need(row, params)
    accum = future_accumulation(row)
    prod, cnt = top_product(user_id)
    pref = mem.get("allocation_preference", "min_risk")
    alloc = allocation_max_return(row) if pref == "max_return" else allocation_min_risk(row, params)
    surplus = val(row, "income", 0) - val(row, "expense", 0)
    retire_age = f"{info['retire_years']}岁" + (f"{info['retire_month_rem']}个月" if info["retire_month_rem"] else "")
    return (
        f"1. 基本情况\n"
        f"客户ID：{user_id}，年龄：{info['age']}岁，性别：{info['gender']}，风险评级：{val(row, 'risk', 'R3')}。"
        f"当前净资产：{int(round(val(row, 'asset', 0)))}元，每月结余：{int(round(surplus))}元。"
        f"每月退休金：{int(round(val(row, 'pension', 0)))}元，企业年金：{int(round(val(row, 'annuity', 0)))}元。\n"
        f"2. 基本假设\n"
        f"预期寿命{params.get('life_expectancy', DEFAULT_LIFE_EXPECTANCY)}岁，长期通胀率2%，默认投资回报率2%，退休年龄约{retire_age}。\n"
        f"3. 养老目标\n"
        f"{mem.get('goal', '退休后维持当前消费水平对应的购买力')}，刚退休时每月约需{int(round(need['monthly_expense_at_retire']))}元。\n"
        f"4. 退休后财富需求测算\n"
        f"退休后预计生活资金现值约{int(round(need['living_pv']))}元，养老金可支撑约{int(round(need['pension_pv']))}元，"
        f"仍需通过投资积累覆盖约{int(round(need['gap']))}元；按默认定期存款测算，退休时预计可积累{int(round(accum))}元。\n"
        f"5. 产品偏好\n"
        f"根据历史行为，客户最偏好的产品类型为{prod}" + (f"（约{cnt}次相关行为）" if cnt else "") + "。\n"
        f"6. 资产配置方式与具体方案\n"
        f"建议采用{'收益最大化' if pref == 'max_return' else '满足养老需求基础上最小化风险波动'}方案：{alloc}。\n"
        f"7. 其他建议\n"
        f"建议客户经理定期复核收入、支出、风险评级和寿命预期变化；若后续风险承受能力提升，可在合规范围内提高长期收益型产品占比。"
    )


def answer(text):
    d = db()
    uid = user_id_from(text)
    params = parse_params(text)

    if re.search(r"多少客户.*年龄.*?(\d+)\s*岁.*?(?:及以上|以上|不低于|大于等于)", text):
        age = float(re.search(r"年龄.*?(\d+)\s*岁", text).group(1))
        return f"{count_age_ge(age)} 个"

    if "平均年龄" in text:
        product = "权益类产品" if "权益" in text else ("现金理财" if "现金" in text else "固收+产品")
        min_count = int(number_before(text, "次", 1) or 1)
        return f"{int(round(avg_age_for_product(product, min_count, '浏览' in text)))} 岁"

    if not uid:
        return "请提供客户ID，例如 V500001。"

    row = d.user(uid)
    if not row:
        return f"未查询到客户 {uid} 的信息。"

    if "建议书" in text or "养老规划" in text and "生成" in text:
        return report(uid, row)

    if "结余" in text:
        return _money(val(row, "income", 0) - val(row, "expense", 0))

    if "行为最多" in text or "偏好" in text and "产品" in text:
        prod, _ = top_product(uid)
        return prod

    if "未来一个星期" in text or "未来一周" in text or "最可能购买" in text:
        prod, _ = top_product(uid, only_buy=True)
        return prod

    if "距离退休" in text or "还有多久退休" in text:
        m = retirement_info(row)["months_left"]
        return f"{m // 12} 年 {m % 12} 个月"

    if "每月需要支出" in text or "退休时月支出" in text or "刚退休时" in text:
        return _money(future_expense(row, params))

    if "最低需要积攒" in text or "最低需要准备" in text or "养老金缺口" in text:
        return _money(living_need(row, params)["gap"])

    if "可以积攒" in text or "能积攒" in text or "退休时可积累" in text:
        rate = DEFAULT_RETURN
        for name, info in PRODUCTS.items():
            if name in text:
                rate = info["rate"]
        return _money(future_accumulation(row, rate))

    if "能否达成" in text or "如何调整" in text or "能不能达成" in text:
        need = living_need(row, params)["gap"]
        rate = PRODUCTS["定期存款"]["rate"] if "定期" in text else DEFAULT_RETURN
        fv = future_accumulation(row, rate)
        if fv + 1e-6 >= need:
            return f"能，预计退休时可积累{int(round(fv))}元，高于所需{int(round(need))}元。"
        name, best_fv, _ = recommend_adjustment(row, params)
        return f"不能，缺口约{int(round(need - fv))}元，建议调整为{name}，预计退休时可积累约{int(round(best_fv))}元。"

    if "寿命" in text and ("增加" in text or "延长" in text or "配置" in text):
        return "年金险"

    if "收益最大化" in text or "投资收益最大" in text:
        return allocation_max_return(row)

    if "最小化风险" in text or "风险波动" in text:
        return allocation_min_risk(row, params)

    field = basic_field_answer(text, row)
    if field is not None:
        return field

    return report(uid, row)


def run(inf):
    text = str(inf or "").strip()
    uid = user_id_from(text)
    try:
        res = answer(text)
        remember(text, uid)
        return res
    except Exception as exc:
        return f"暂时无法完成该问题，请检查数据表字段；错误信息：{type(exc).__name__}"


if __name__ == "__main__":
    while True:
        try:
            q = input("Q: ")
        except EOFError:
            break
        print(run(q))
