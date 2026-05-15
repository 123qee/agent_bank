import pandas as pd
import numpy as np
import os
import json
import time
import random
from collections import Counter, defaultdict

print("🚀 启动[v2-CF/旋转/诊断]重排引擎 ...")

DEBUG_LOG_PATH = '/Users/mac/Desktop/first/.cursor/debug-8ab6b0.log'
DEBUG_SESSION_ID = '8ab6b0'

# region agent log
def agent_log(hypothesis_id, message, data=None, location='notebook'):
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": "full-debug-v2",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion


# =========================
# 1. 加载训练数据
# =========================
train_df = pd.read_csv('/work/data/task1/dataset_train.csv')
train_df['acs_tm'] = pd.to_datetime(train_df['acs_tm'])

cols = ['action_typ', 'prod_typ', 'prod_sub_typ', 'rsk_lvl']
for col in cols:
    train_df[col] = train_df[col].fillna('unknown').astype(str)

train_df['item'] = (
    train_df['action_typ'] + '|' +
    train_df['prod_typ'] + '|' +
    train_df['prod_sub_typ'] + '|' +
    train_df['rsk_lvl']
)

rsk_map = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5}

# region agent log
agent_log("H3-H4", "train data loaded", {
    "rows": len(train_df),
    "users": train_df['user_id'].nunique(),
    "min_time": str(train_df['acs_tm'].min()),
    "max_time": str(train_df['acs_tm'].max()),
    "item_nulls": int(train_df['item'].isna().sum()),
    "bad_item_format": int((train_df['item'].str.count('\\|') != 3).sum())
}, "notebook:load_train")
# endregion


# =========================
# 2. 特征构建函数
# =========================
def build_features(hist_df, tag='unknown'):
    t0 = time.time()
    print(f"[{tag}] build_features 启动 rows={len(hist_df)} users={hist_df['user_id'].nunique()}")

    # region agent log
    agent_log("H4", "build_features enter", {
        "tag": tag,
        "rows": len(hist_df),
        "users": hist_df['user_id'].nunique() if len(hist_df) else 0,
        "min_time": str(hist_df['acs_tm'].min()) if len(hist_df) else None,
        "max_time": str(hist_df['acs_tm'].max()) if len(hist_df) else None
    }, "notebook:build_features:start")
    # endregion

    item_cnt = hist_df['item'].value_counts()
    if item_cnt.empty:
        raise ValueError(f"{tag}: hist_df 为空或没有可用 item")

    global_top_items = item_cnt.index.tolist()
    max_item_cnt = int(item_cnt.iloc[0])

    item_pop_score = {
        k: float(np.log1p(v) / np.log1p(max_item_cnt))
        for k, v in item_cnt.items()
    }

    valid_combos = set(global_top_items)

    # (p, s, r) -> 全局合法 action_typ 集合，用于动作旋转
    valid_actions_for_psr = defaultdict(set)
    for itm in valid_combos:
        parts = itm.split('|')
        if len(parts) == 4:
            a, p, s, r = parts
            valid_actions_for_psr[(p, s, r)].add(a)

    # (p, s, r) -> action_typ 在该三元组内的全局概率
    psr_action_prob = defaultdict(dict)
    for itm, cnt in item_cnt.items():
        parts = itm.split('|')
        if len(parts) == 4:
            a, p, s, r = parts
            psr_action_prob[(p, s, r)][a] = int(cnt)
    for psr, m in psr_action_prob.items():
        total = sum(m.values()) or 1
        for a in m:
            m[a] = m[a] / total

    # 全局时间衰减（更陡，突出最近）
    recent_df = hist_df.copy()
    max_day = recent_df['acs_tm'].max()
    recent_df['days_ago'] = (max_day - recent_df['acs_tm']).dt.days
    recent_df['rec_weight'] = np.exp(-recent_df['days_ago'] / 3.0)
    item_recent_series = recent_df.groupby('item')['rec_weight'].sum()
    item_recent_score = (item_recent_series / item_recent_series.max()).to_dict()

    # 用户最近一次 rsk_lvl（字符串 + 数值）
    user_last_rsk_str = (
        hist_df.sort_values('acs_tm')
        .groupby('user_id')['rsk_lvl']
        .last()
        .to_dict()
    )
    user_last_rsk = {u: rsk_map.get(r, 1) for u, r in user_last_rsk_str.items()}

    # 用户历史 item 列表（按时间从近到远）
    user_hist_items = (
        hist_df.sort_values('acs_tm', ascending=False)
        .groupby('user_id')['item']
        .apply(list)
        .to_dict()
    )

    user_unique_recent_items = {
        u: list(dict.fromkeys([x for x in items if isinstance(x, str) and x.count('|') == 3]))
        for u, items in user_hist_items.items()
    }

    # 最近时间窗口
    day_1d = max_day - pd.Timedelta(days=1)
    day_3d = max_day - pd.Timedelta(days=3)
    day_7d = max_day - pd.Timedelta(days=7)

    user_last_1day_items = (
        hist_df[hist_df['acs_tm'] >= day_1d]
        .groupby('user_id')['item'].apply(set).to_dict()
    )
    user_last_3day_items = (
        hist_df[hist_df['acs_tm'] >= day_3d]
        .groupby('user_id')['item'].apply(set).to_dict()
    )
    user_last_7day_items = (
        hist_df[hist_df['acs_tm'] >= day_7d]
        .groupby('user_id')['item'].apply(set).to_dict()
    )

    # 用户在最近三天内每个 item 出现次数
    user_last_3day_item_cnt = (
        hist_df[hist_df['acs_tm'] >= day_3d]
        .groupby(['user_id', 'item']).size()
    )
    user_last_3day_item_cnt_dict = {
        u: dict(g.droplevel(0)) for u, g in user_last_3day_item_cnt.groupby(level=0)
    }
    user_last_1day_item_cnt = (
        hist_df[hist_df['acs_tm'] >= day_1d]
        .groupby(['user_id', 'item']).size()
    )
    user_last_1day_item_cnt_dict = {
        u: dict(g.droplevel(0)) for u, g in user_last_1day_item_cnt.groupby(level=0)
    }

    # 用户维度偏好
    user_action_pref = hist_df.groupby('user_id')['action_typ'].apply(lambda x: Counter(x)).to_dict()
    user_prod_pref = hist_df.groupby('user_id')['prod_typ'].apply(lambda x: Counter(x)).to_dict()
    user_sub_pref = hist_df.groupby('user_id')['prod_sub_typ'].apply(lambda x: Counter(x)).to_dict()
    user_rsk_pref = hist_df.groupby('user_id')['rsk_lvl'].apply(lambda x: Counter(x)).to_dict()

    recent_3d_df = hist_df[hist_df['acs_tm'] >= day_3d]
    user_recent_action_pref = recent_3d_df.groupby('user_id')['action_typ'].apply(lambda x: Counter(x)).to_dict()
    user_recent_prod_pref = recent_3d_df.groupby('user_id')['prod_typ'].apply(lambda x: Counter(x)).to_dict()
    user_recent_sub_pref = recent_3d_df.groupby('user_id')['prod_sub_typ'].apply(lambda x: Counter(x)).to_dict()

    # 用户 (p, s, r) 三元组集合（向量化避免 iterrows）
    psr_tmp = hist_df[['user_id', 'prod_typ', 'prod_sub_typ', 'rsk_lvl']].drop_duplicates()
    psr_tmp['psr'] = list(zip(psr_tmp['prod_typ'], psr_tmp['prod_sub_typ'], psr_tmp['rsk_lvl']))
    user_psr_set = psr_tmp.groupby('user_id')['psr'].apply(set).to_dict()

    pension_users = set(
        hist_df[hist_df['prod_sub_typ'].str.contains('养老', na=False)]['user_id'].unique()
    )

    # 用户复购倾向（最高 item 频次 / 历史总长度），高=偏稳定
    user_repeat_ratio = {}
    for u, items in user_hist_items.items():
        if items:
            c = Counter(items)
            user_repeat_ratio[u] = max(c.values()) / len(items)
        else:
            user_repeat_ratio[u] = 0.0

    # ===== Item-CF: 基于用户共现的 item-item 余弦相似度 =====
    print(f"[{tag}] 构建 Item-CF 共现矩阵 ...")
    try:
        from scipy.sparse import csr_matrix
        unique_items_list = list(valid_combos)
        item_idx = {it: i for i, it in enumerate(unique_items_list)}
        users_list = list(user_unique_recent_items.keys())
        u_idx_map = {u: i for i, u in enumerate(users_list)}

        rows, cols_a = [], []
        for u, items in user_unique_recent_items.items():
            ui = u_idx_map[u]
            for itm in items:
                ii = item_idx.get(itm)
                if ii is not None:
                    rows.append(ui)
                    cols_a.append(ii)

        n_u = len(users_list)
        n_i = len(unique_items_list)
        B = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols_a)),
            shape=(n_u, n_i)
        )
        # 二值化（同一用户同一 item 只算一次）
        B.data = np.minimum(B.data, 1.0)

        cooc = (B.T @ B).toarray().astype(np.float32)
        deg = np.diag(cooc).copy()
        np.fill_diagonal(cooc, 0.0)
        # 余弦相似度
        denom = np.sqrt(deg[:, None] * deg[None, :]) + 1e-8
        item_sim = (cooc / denom).astype(np.float32)
        item_cf_available = True
        print(f"[{tag}] Item-CF OK: items={n_i} users={n_u} nnz={int(B.nnz)}")
    except Exception as e:
        print(f"[{tag}] Item-CF 构建失败 -> {e}")
        item_sim = None
        item_idx = {}
        item_cf_available = False

    feat = {
        'global_top_items': global_top_items,
        'item_pop_score': item_pop_score,
        'item_recent_score': item_recent_score,
        'valid_combos': valid_combos,
        'valid_actions_for_psr': dict(valid_actions_for_psr),
        'psr_action_prob': dict(psr_action_prob),

        'user_last_rsk': user_last_rsk,
        'user_last_rsk_str': user_last_rsk_str,
        'user_hist_items': user_hist_items,
        'user_unique_recent_items': user_unique_recent_items,
        'user_last_1day_items': user_last_1day_items,
        'user_last_3day_items': user_last_3day_items,
        'user_last_7day_items': user_last_7day_items,
        'user_last_3day_item_cnt': user_last_3day_item_cnt_dict,
        'user_last_1day_item_cnt': user_last_1day_item_cnt_dict,
        'user_action_pref': user_action_pref,
        'user_prod_pref': user_prod_pref,
        'user_sub_pref': user_sub_pref,
        'user_rsk_pref': user_rsk_pref,
        'user_recent_action_pref': user_recent_action_pref,
        'user_recent_prod_pref': user_recent_prod_pref,
        'user_recent_sub_pref': user_recent_sub_pref,
        'user_psr_set': user_psr_set,
        'user_repeat_ratio': user_repeat_ratio,
        'pension_users': pension_users,

        'item_sim': item_sim,
        'item_idx': item_idx,
        'item_cf_available': item_cf_available,
    }

    # region agent log
    agent_log("H4", "build_features exit", {
        "tag": tag,
        "global_items": len(global_top_items),
        "users_with_hist": len(user_hist_items),
        "pension_users": len(pension_users),
        "item_cf_available": item_cf_available,
        "elapsed_s": round(time.time() - t0, 1)
    }, "notebook:build_features:end")
    # endregion

    print(f"[{tag}] build_features 完成耗时 {time.time() - t0:.1f}s")
    return feat


# =========================
# 3. 单用户预测函数
# =========================
def predict_for_user(user, feat, top_global_n=120, return_debug=False):
    user_items = feat['user_hist_items'].get(user, [])
    unique_recent_items = feat['user_unique_recent_items'].get(user, [])
    last_1day = feat['user_last_1day_items'].get(user, set())
    last_3day = feat['user_last_3day_items'].get(user, set())
    last_7day = feat['user_last_7day_items'].get(user, set())
    last_3day_cnt = feat['user_last_3day_item_cnt'].get(user, {})
    last_1day_cnt = feat['user_last_1day_item_cnt'].get(user, {})
    user_psr = feat['user_psr_set'].get(user, set())
    valid_actions_for_psr = feat['valid_actions_for_psr']
    psr_action_prob = feat['psr_action_prob']

    action_pref = feat['user_action_pref'].get(user, Counter())
    prod_pref = feat['user_prod_pref'].get(user, Counter())
    sub_pref = feat['user_sub_pref'].get(user, Counter())
    rsk_pref = feat['user_rsk_pref'].get(user, Counter())

    r_action_pref = feat['user_recent_action_pref'].get(user, Counter())
    r_prod_pref = feat['user_recent_prod_pref'].get(user, Counter())
    r_sub_pref = feat['user_recent_sub_pref'].get(user, Counter())

    u_rsk_str = feat['user_last_rsk_str'].get(user, 'R1')
    u_rsk = feat['user_last_rsk'].get(user, 1)
    is_pen_fan = user in feat['pension_users']
    valid_combos = feat['valid_combos']
    repeat_ratio = feat['user_repeat_ratio'].get(user, 0.0)

    # ---- Item-CF: 计算该用户对每个 item 的 CF 加权得分 ----
    cf_score_map = {}
    if feat['item_cf_available'] and unique_recent_items:
        item_sim = feat['item_sim']
        item_idx = feat['item_idx']
        hist_indices = []
        hist_weights = []
        for i, itm in enumerate(unique_recent_items[:30]):
            idx = item_idx.get(itm)
            if idx is not None:
                hist_indices.append(idx)
                hist_weights.append(np.exp(-i / 5.0))  # 越靠近最近，权重越大
        if hist_indices:
            hist_arr = np.array(hist_indices, dtype=np.int64)
            w_arr = np.array(hist_weights, dtype=np.float32).reshape(-1, 1)
            cf_vec = (item_sim[hist_arr] * w_arr).sum(axis=0)
            mx = cf_vec.max()
            if mx > 0:
                cf_vec = cf_vec / mx
            # 只对 top 200 个 CF 候选生效（避免每个 item 都给 cf 分）
            top_cf_idx = np.argsort(-cf_vec)[:200]
            inv_idx = {v: k for k, v in item_idx.items()}
            for i in top_cf_idx:
                if cf_vec[i] > 0:
                    cf_score_map[inv_idx[i]] = float(cf_vec[i])

    # ---- 候选集 ----
    candidates = []
    seen = set()

    def add_cand(itm):
        if itm in seen:
            return
        if not isinstance(itm, str) or itm.count('|') != 3:
            return
        if itm not in valid_combos:
            return
        candidates.append(itm)
        seen.add(itm)

    # 1. 用户历史 unique items
    for itm in unique_recent_items:
        add_cand(itm)

    # 2. Action 旋转：对用户每个历史 (p, s, r)，补齐所有合法 action_typ
    for psr in user_psr:
        for a in valid_actions_for_psr.get(psr, set()):
            add_cand(f"{a}|{psr[0]}|{psr[1]}|{psr[2]}")

    # 3. Item-CF top 候选
    for itm in cf_score_map.keys():
        add_cand(itm)

    # 4. 用户偏好维度笛卡尔积（rsk 强制限定在用户偏好/最近 rsk）
    if action_pref and prod_pref and sub_pref:
        def merge_topk(short_pref, long_pref, k):
            merged = Counter()
            for kk, vv in short_pref.items():
                merged[kk] += 2.0 * vv
            for kk, vv in long_pref.items():
                merged[kk] += 1.0 * vv
            return [x for x, _ in merged.most_common(k)]

        top_actions = merge_topk(r_action_pref, action_pref, 3)
        top_prods = merge_topk(r_prod_pref, prod_pref, 3)
        top_subs = merge_topk(r_sub_pref, sub_pref, 4)
        top_rsks_list = [r for r, _ in rsk_pref.most_common(2)]
        if u_rsk_str not in top_rsks_list:
            top_rsks_list.insert(0, u_rsk_str)
        top_rsks = top_rsks_list[:2]

        for a in top_actions:
            for p in top_prods:
                for s in top_subs:
                    for r in top_rsks:
                        add_cand(f"{a}|{p}|{s}|{r}")

    # 5. 全局热门（仅当候选不足时启用大量兜底）
    for itm in feat['global_top_items'][:top_global_n]:
        add_cand(itm)

    # ---- 打分 ----
    item_freq = Counter(user_items)
    max_user_freq = max(item_freq.values()) if item_freq else 1
    max_action = max(action_pref.values()) if action_pref else 1
    max_prod = max(prod_pref.values()) if prod_pref else 1
    max_sub = max(sub_pref.values()) if sub_pref else 1
    max_rsk = max(rsk_pref.values()) if rsk_pref else 1
    max_r_action = max(r_action_pref.values()) if r_action_pref else 1
    max_r_prod = max(r_prod_pref.values()) if r_prod_pref else 1
    max_r_sub = max(r_sub_pref.values()) if r_sub_pref else 1
    max_3d_cnt = max(last_3day_cnt.values()) if last_3day_cnt else 1
    max_1d_cnt = max(last_1day_cnt.values()) if last_1day_cnt else 1

    pos_map = {itm: i for i, itm in enumerate(unique_recent_items)}

    debug_rows = []
    scored = []

    for itm in candidates:
        action_typ, prod_typ, prod_sub_typ, rsk_lvl = itm.split('|')
        itm_rsk = rsk_map.get(rsk_lvl, 1)

        user_freq_score = item_freq.get(itm, 0) / max_user_freq
        global_score = feat['item_pop_score'].get(itm, 0)
        recent_score = feat['item_recent_score'].get(itm, 0)
        action_score = action_pref.get(action_typ, 0) / max_action
        prod_score = prod_pref.get(prod_typ, 0) / max_prod
        sub_score = sub_pref.get(prod_sub_typ, 0) / max_sub
        rsk_score = rsk_pref.get(rsk_lvl, 0) / max_rsk
        r_action_score = r_action_pref.get(action_typ, 0) / max_r_action
        r_prod_score = r_prod_pref.get(prod_typ, 0) / max_r_prod
        r_sub_score = r_sub_pref.get(prod_sub_typ, 0) / max_r_sub
        win3_score = last_3day_cnt.get(itm, 0) / max_3d_cnt
        win1_score = last_1day_cnt.get(itm, 0) / max_1d_cnt
        cf_score = cf_score_map.get(itm, 0.0)
        psr_a_prob = psr_action_prob.get((prod_typ, prod_sub_typ, rsk_lvl), {}).get(action_typ, 0.0)
        user_has_psr = 1.0 if (prod_typ, prod_sub_typ, rsk_lvl) in user_psr else 0.0

        score = 0.0
        # 用户实际历史重复（最强）
        score += 4.5 * user_freq_score
        score += 7.0 * win1_score          # 最近一天命中频次
        score += 4.0 * win3_score          # 最近三天命中频次
        # 用户走过的 (p,s,r) 是高确定区，配合动作旋转
        score += 3.0 * user_has_psr * (1.0 + 1.5 * psr_a_prob)  # 命中 (p,s,r)+常见 action
        # CF 加成
        score += 3.0 * cf_score
        # 全局信号
        score += 0.6 * global_score
        score += 1.4 * recent_score
        # 用户长期 + 短期 维度偏好
        score += 0.8 * action_score
        score += 1.0 * prod_score
        score += 1.6 * sub_score
        score += 0.4 * rsk_score
        score += 1.2 * r_action_score
        score += 1.4 * r_prod_score
        score += 2.0 * r_sub_score

        # 最近时间窗口出现过（绝对加分）
        if itm in last_1day:
            score += 7.0
        elif itm in last_3day:
            score += 3.0
        elif itm in last_7day:
            score += 1.2

        # 用户最近 N 个 unique item 位置加成
        pos = pos_map.get(itm)
        if pos is not None:
            if pos < 3:
                score += 3.0
            elif pos < 8:
                score += 1.5
            elif pos < 15:
                score += 0.6

        # ----- 强 rsk 错配惩罚（用户属性短期稳定） -----
        if itm_rsk == u_rsk:
            score *= 1.35
        else:
            # 用户是否真的在此 rsk_lvl 有过行为
            if rsk_lvl in rsk_pref and rsk_pref[rsk_lvl] >= 3:
                score *= 0.75
            else:
                diff = abs(itm_rsk - u_rsk)
                if diff == 1:
                    score *= 0.45
                elif diff == 2:
                    score *= 0.20
                else:
                    score *= 0.08

        # unknown rsk 直接重罚（用户基本不会触发 unknown 行为）
        if rsk_lvl == 'unknown' and u_rsk_str != 'unknown':
            score *= 0.15
        if prod_sub_typ == 'unknown' and any(s != 'unknown' for s in sub_pref):
            score *= 0.5

        # 养老倾向用户：温和加成
        if is_pen_fan and '养老' in prod_sub_typ:
            score *= 1.3

        # 稳定用户（高 repeat_ratio）：放大用户历史信号
        if repeat_ratio > 0.05 and itm in last_3day:
            score *= 1.08

        scored.append((itm, score))
        if return_debug:
            debug_rows.append({
                "itm": itm, "score": round(score, 3),
                "u_freq": round(user_freq_score, 2),
                "win1": round(win1_score, 2),
                "win3": round(win3_score, 2),
                "cf": round(cf_score, 2),
                "in_1d": itm in last_1day,
                "in_3d": itm in last_3day,
                "psr_a_prob": round(psr_a_prob, 2),
                "user_has_psr": int(user_has_psr),
                "rsk_match": itm_rsk == u_rsk,
            })

    preds = [x[0] for x in sorted(scored, key=lambda x: x[1], reverse=True)]

    # 兜底补齐到 20
    seen2 = set(preds)
    for itm in feat['global_top_items']:
        if len(preds) >= 20:
            break
        if itm not in seen2 and isinstance(itm, str) and itm.count('|') == 3:
            preds.append(itm)
            seen2.add(itm)

    preds = preds[:20]

    if return_debug:
        debug_rows.sort(key=lambda r: -r['score'])
        return preds, debug_rows[:30]
    return preds


# =========================
# 4. 本地验证 + 大量诊断日志
# =========================
print("📊 本地验证：用 3/1-3/31 预测 4/1 ...")

hist_part = train_df[train_df['acs_tm'] < '2025-04-01'].copy()
val_part = train_df[train_df['acs_tm'] >= '2025-04-01'].copy()

# region agent log
agent_log("H4", "validation split created", {
    "hist_rows": len(hist_part),
    "hist_users": hist_part['user_id'].nunique(),
    "val_rows": len(val_part),
    "val_users": val_part['user_id'].nunique(),
    "hist_min_time": str(hist_part['acs_tm'].min()) if len(hist_part) else None,
    "hist_max_time": str(hist_part['acs_tm'].max()) if len(hist_part) else None,
    "val_min_time": str(val_part['acs_tm'].min()) if len(val_part) else None,
    "val_max_time": str(val_part['acs_tm'].max()) if len(val_part) else None
}, "notebook:validation_split")
# endregion

val_truth = val_part.groupby('user_id')['item'].apply(set).to_dict()
feat_val = build_features(hist_part, tag='validation')

scores = []
eval_users = list(val_truth.keys())

# 详细诊断容器
recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
hit_positions_all = []
total_truths = 0
total_hits = 0
miss_in_history = 0
miss_in_candidates_only = 0
miss_total = 0

truth_size_dist = Counter()
hits_by_truth_size = defaultdict(lambda: [0, 0])  # truth_size -> [sum_ndcg, count]

# action_typ / prod_typ / sub / rsk 维度命中率（检查我们最常错在哪个维度）
attr_match_counter = {
    "action_only_match": 0,
    "prod_only_match": 0,
    "sub_only_match": 0,
    "rsk_only_match": 0,
    "action_prod_sub_match_but_rsk_wrong": 0,
    "all_match": 0,
    "none_match": 0,
}

# 对前 N 个用户额外采样用于打印
sample_users_for_dump = set(random.Random(42).sample(eval_users, min(5, len(eval_users))))
sample_dump_records = []

t_val_start = time.time()
for n, user in enumerate(eval_users, start=1):
    if n % 10000 == 0:
        elapsed = time.time() - t_val_start
        print(f"已验证 {n}/{len(eval_users)} 用户 用时 {elapsed:.1f}s")

    true_items = val_truth[user]
    truth_size_dist[min(len(true_items), 20)] += 1
    total_truths += len(true_items)

    want_debug = user in sample_users_for_dump
    if want_debug:
        preds, dbg = predict_for_user(user, feat_val, return_debug=True)
    else:
        preds = predict_for_user(user, feat_val)
        dbg = None

    pred_set = set(preds)

    # NDCG
    dcg = 0.0
    user_hit_pos = []
    for i, p in enumerate(preds):
        if p in true_items:
            dcg += 1.0 / np.log2(i + 2)
            user_hit_pos.append(i + 1)
            total_hits += 1
            hit_positions_all.append(i + 1)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), 20)))
    user_ndcg = dcg / idcg if idcg > 0 else 0.0
    scores.append(user_ndcg)

    hits_by_truth_size[min(len(true_items), 20)][0] += user_ndcg
    hits_by_truth_size[min(len(true_items), 20)][1] += 1

    for k in recall_at_k:
        if any(p in true_items for p in preds[:k]):
            recall_at_k[k] += 1

    # miss 分析
    hist_set = set(feat_val['user_unique_recent_items'].get(user, []))
    cand_set_proxy = hist_set  # 真实候选集计算贵；用历史代理
    missed = true_items - pred_set
    for m in missed:
        miss_total += 1
        if m in hist_set:
            miss_in_history += 1

    # 4 维属性匹配分析（用于诊断"我们错在哪一维"）
    # 对每个 truth item，看 preds 中是否有"差一个维度"的近邻
    truth_split = [t.split('|') for t in true_items]
    pred_split = [p.split('|') for p in preds]
    for ts in truth_split:
        if len(ts) != 4:
            continue
        t_a, t_p, t_s, t_r = ts
        all_match = False
        psr_match_but_rsk_wrong = False
        any_action_match = False
        any_prod_match = False
        any_sub_match = False
        any_rsk_match = False
        for ps in pred_split:
            if len(ps) != 4:
                continue
            p_a, p_p, p_s, p_r = ps
            if p_a == t_a and p_p == t_p and p_s == t_s and p_r == t_r:
                all_match = True
                break
            if p_a == t_a and p_p == t_p and p_s == t_s and p_r != t_r:
                psr_match_but_rsk_wrong = True
            if p_a == t_a: any_action_match = True
            if p_p == t_p: any_prod_match = True
            if p_s == t_s: any_sub_match = True
            if p_r == t_r: any_rsk_match = True
        if all_match:
            attr_match_counter["all_match"] += 1
        else:
            if psr_match_but_rsk_wrong:
                attr_match_counter["action_prod_sub_match_but_rsk_wrong"] += 1
            if any_action_match: attr_match_counter["action_only_match"] += 1
            if any_prod_match: attr_match_counter["prod_only_match"] += 1
            if any_sub_match: attr_match_counter["sub_only_match"] += 1
            if any_rsk_match: attr_match_counter["rsk_only_match"] += 1
            if not (any_action_match or any_prod_match or any_sub_match or any_rsk_match):
                attr_match_counter["none_match"] += 1

    if want_debug:
        sample_dump_records.append({
            "user": user,
            "ndcg": round(user_ndcg, 3),
            "truth_size": len(true_items),
            "truth_items": sorted(true_items)[:10],
            "preds_top10": preds[:10],
            "hits_positions": user_hit_pos,
            "hist_top5": feat_val['user_unique_recent_items'].get(user, [])[:5],
            "last_rsk": feat_val['user_last_rsk_str'].get(user, '?'),
            "scoring_top10": dbg[:10] if dbg else None,
            "missed_items": list(true_items - pred_set)[:5],
            "missed_in_history": [m for m in (true_items - pred_set) if m in hist_set][:5],
        })

score_mean = float(np.mean(scores)) if scores else 0.0
n_eval = len(eval_users)

# ===== 详细汇报 =====
print("\n" + "=" * 60)
print(f"🎯 本地模拟 NDCG@20: {score_mean:.4f}")
print("=" * 60)
print(f"评估用户: {n_eval}")
print(f"真实 item 总数: {total_truths} | 命中总数: {total_hits} | 总体命中率: {total_hits/max(total_truths,1):.4f}")
print(f"真实 item / 用户: 平均={total_truths/max(n_eval,1):.2f}")
print()
print("Recall@K (用户层至少命中一个 truth 的比例):")
for k in [1, 3, 5, 10, 20]:
    print(f"  Recall@{k}: {recall_at_k[k]/n_eval:.4f}  ({recall_at_k[k]}/{n_eval})")
print()

if hit_positions_all:
    hp = np.array(hit_positions_all)
    print("命中位置分布 (item 级):")
    for k in [1, 2, 3, 5, 10, 15, 20]:
        cnt = int(np.sum(hp <= k))
        print(f"  在 top{k} 内命中: {cnt} ({cnt/total_truths:.2%})")
    print(f"  平均命中位置: {hp.mean():.2f}")
    print(f"  中位命中位置: {int(np.median(hp))}")

print()
print(f"Miss 分析: 总 miss={miss_total}")
if miss_total > 0:
    print(f"  miss 在用户历史中(真实回头但被我们排除): {miss_in_history} ({miss_in_history/miss_total:.2%})")
    print(f"  miss 未在用户历史中(冷启): {miss_total - miss_in_history} ({(miss_total-miss_in_history)/miss_total:.2%})")

print()
print("Truth set 大小分布:")
for sz in sorted(truth_size_dist.keys())[:10]:
    cnt = truth_size_dist[sz]
    avg_ndcg = hits_by_truth_size[sz][0] / max(hits_by_truth_size[sz][1], 1)
    print(f"  size={sz:>3d}: 用户数={cnt:>6d}  avg_NDCG={avg_ndcg:.4f}")

print()
print("4 维属性诊断 (对每个 truth item，preds 中至少一个在该维度匹配的次数):")
for k, v in attr_match_counter.items():
    pct = v / max(total_truths, 1)
    print(f"  {k:>45s}: {v:>8d}  ({pct:.2%})")
print()

print("=" * 60)
print("🔬 随机采样 5 个用户的预测详情 (帮助调权重):")
print("=" * 60)
for rec in sample_dump_records:
    print(f"\n👤 {rec['user']}  NDCG={rec['ndcg']}  truth_size={rec['truth_size']}  last_rsk={rec['last_rsk']}")
    print(f"  truth (前10): {rec['truth_items']}")
    print(f"  preds top10 : {rec['preds_top10']}")
    print(f"  命中位置    : {rec['hits_positions']}")
    print(f"  历史 top5   : {rec['hist_top5']}")
    if rec['missed_items']:
        print(f"  丢失 items  : {rec['missed_items']}")
        if rec['missed_in_history']:
            print(f"  ↪ 其中本来就在历史里却被排除的: {rec['missed_in_history']}")
    if rec['scoring_top10']:
        print(f"  分数 top10 拆解:")
        for r in rec['scoring_top10']:
            print(f"    {r}")

print()
print(f"⏱  本地验证耗时 {time.time() - t_val_start:.1f}s")

# region agent log
agent_log("H1-H4", "validation finished", {
    "eval_users": n_eval,
    "score_mean": score_mean,
    "recall_at_1": recall_at_k[1] / n_eval,
    "recall_at_5": recall_at_k[5] / n_eval,
    "recall_at_10": recall_at_k[10] / n_eval,
    "recall_at_20": recall_at_k[20] / n_eval,
    "miss_in_history_ratio": miss_in_history / max(miss_total, 1),
    "attr_match_counter": attr_match_counter,
}, "notebook:validation_end")
# endregion


# =========================
# 5. 生成最终提交
# =========================
print("\n📝 生成最终提交文件 ...")

test_df = pd.read_csv('/work/data/task1/dataset_test.csv')
test_df['acs_tm'] = pd.to_datetime(test_df['acs_tm'])

for col in cols:
    test_df[col] = test_df[col].fillna('unknown').astype(str)

test_df['item'] = (
    test_df['action_typ'] + '|' +
    test_df['prod_typ'] + '|' +
    test_df['prod_sub_typ'] + '|' +
    test_df['rsk_lvl']
)

# region agent log
agent_log("H3-H5", "test data loaded", {
    "rows": len(test_df),
    "users": test_df['user_id'].nunique(),
    "min_time": str(test_df['acs_tm'].min()),
    "max_time": str(test_df['acs_tm'].max())
}, "notebook:load_test")
# endregion

final_hist = pd.concat([
    train_df[train_df['acs_tm'] < '2025-04-01'],
    test_df
], ignore_index=True)

feat_final = build_features(final_hist, tag='final')

rows_out = []
test_users = test_df['user_id'].drop_duplicates().tolist()

t_sub_start = time.time()
for n, user in enumerate(test_users, start=1):
    if n % 10000 == 0:
        print(f"已生成 {n}/{len(test_users)} 用户 用时 {time.time()-t_sub_start:.1f}s")

    preds = predict_for_user(user, feat_final)

    if len(preds) != 20:
        agent_log("H5", "prediction length not 20", {
            "user": user,
            "pred_len": len(preds)
        }, "notebook:submit_loop")

    for idx, itm in enumerate(preds, start=1):
        action_typ, prod_typ, prod_sub_typ, rsk_lvl = itm.split('|')
        rows_out.append([user, action_typ, prod_typ, prod_sub_typ, rsk_lvl, idx])

sub = pd.DataFrame(rows_out, columns=[
    'user_id', 'action_typ', 'prod_typ', 'prod_sub_typ', 'rsk_lvl', 'index'
])

cnt = sub.groupby('user_id').size()
agent_log("H5", "submission built", {
    "rows": len(sub),
    "users": sub['user_id'].nunique(),
    "expected_users": len(test_users),
    "cnt_min": int(cnt.min()) if len(cnt) else None,
    "cnt_max": int(cnt.max()) if len(cnt) else None,
    "user_set_match": bool(set(sub['user_id']) == set(test_df['user_id']))
}, "notebook:submission_check")

assert cnt.min() == 20 and cnt.max() == 20, cnt.describe()
assert set(sub['user_id']) == set(test_df['user_id']), "提交用户与测试集用户不一致"

os.makedirs('/work/task1/submit', exist_ok=True)
sub.to_csv('/work/task1/submit/prediction.csv', index=False, encoding='utf-8-sig')

print("✅ prediction.csv 已生成：/work/task1/submit/prediction.csv")
print(sub.head(25))
print(sub.groupby('user_id').size().describe())
print(f"🪵 Debug日志路径: {DEBUG_LOG_PATH}")
