import pandas as pd
import numpy as np
import os
import json
import time
from collections import Counter

print("🚀 启动[增强版]重排引擎 + Debug验证日志...")

DEBUG_LOG_PATH = '/Users/mac/Desktop/first/.cursor/debug-8ab6b0.log'
DEBUG_SESSION_ID = '8ab6b0'

# region agent log
def agent_log(hypothesis_id, message, data=None, location='notebook'):
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": "full-debug-v1",
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
        # region agent log
        agent_log("H4", "item_cnt empty before failure", {
            "tag": tag,
            "rows": len(hist_df)
        }, "notebook:build_features:item_cnt")
        # endregion
        raise ValueError(f"{tag}: hist_df 为空或没有可用 item，请检查时间切分")

    global_top_items = item_cnt.index.tolist()
    max_item_cnt = item_cnt.iloc[0]

    item_pop_score = {
        k: np.log1p(v) / np.log1p(max_item_cnt)
        for k, v in item_cnt.items()
    }

    # 全局合法的 (action, prod, sub, rsk) 组合，用于过滤笛卡尔积扩展
    valid_combos = set(item_cnt.index.tolist())

    recent_df = hist_df.copy()
    max_day = recent_df['acs_tm'].max()
    recent_df['days_ago'] = (max_day - recent_df['acs_tm']).dt.days
    # 更陡的全局时间衰减：临近最后一天的全局热度更突出
    recent_df['rec_weight'] = np.exp(-recent_df['days_ago'] / 3.0)

    item_recent_score_series = recent_df.groupby('item')['rec_weight'].sum()
    item_recent_score = (item_recent_score_series / item_recent_score_series.max()).to_dict()

    # 用户最近的风评等级（用户级稳定属性）
    user_last_rsk_str = (
        hist_df.sort_values('acs_tm')
        .groupby('user_id')['rsk_lvl']
        .last()
        .to_dict()
    )
    user_last_rsk = {
        u: rsk_map.get(r, 1) for u, r in user_last_rsk_str.items()
    }

    user_hist_items = (
        hist_df.sort_values('acs_tm', ascending=False)
        .groupby('user_id')['item']
        .apply(list)
        .to_dict()
    )

    # 用户最近时间窗口下出现过的 item 集合（强复发信号）
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

    # 用户最近窗口的高频 item 频次（用于在窗口内再排序）
    user_last_3day_item_cnt = (
        hist_df[hist_df['acs_tm'] >= day_3d]
        .groupby(['user_id', 'item']).size()
    )
    user_last_3day_item_cnt = {
        u: dict(g.droplevel(0)) for u, g in user_last_3day_item_cnt.groupby(level=0)
    }

    # region agent log
    sample_user = next(iter(user_hist_items), None)
    sample_items = user_hist_items.get(sample_user, [])
    agent_log("H1-H2-H3", "user_hist_items sample before unique", {
        "tag": tag,
        "sample_user": sample_user,
        "items_type": type(sample_items).__name__,
        "items_len": len(sample_items) if hasattr(sample_items, "__len__") else None,
        "first_item_type": type(sample_items[0]).__name__ if sample_items else None,
        "first_item": sample_items[0] if sample_items else None
    }, "notebook:build_features:user_hist_items")
    # endregion

    # 修复点：pandas 当前版本 pd.unique(list) 会 TypeError，改用 dict.fromkeys 保序去重
    user_unique_recent_items = {
        u: list(dict.fromkeys([x for x in items if isinstance(x, str) and x.count('|') == 3]))
        for u, items in user_hist_items.items()
    }

    # region agent log
    bad_hist_items = 0
    for items in list(user_hist_items.values())[:1000]:
        bad_hist_items += sum(1 for x in items if not isinstance(x, str) or x.count('|') != 3)

    agent_log("H1-H2-H3", "user_unique_recent_items built", {
        "tag": tag,
        "users": len(user_unique_recent_items),
        "sample_unique_len": len(user_unique_recent_items.get(sample_user, [])) if sample_user else 0,
        "bad_hist_items_in_first_1000_users": bad_hist_items
    }, "notebook:build_features:user_unique_recent_items")
    # endregion

    user_action_pref = hist_df.groupby('user_id')['action_typ'].apply(lambda x: Counter(x)).to_dict()
    user_prod_pref = hist_df.groupby('user_id')['prod_typ'].apply(lambda x: Counter(x)).to_dict()
    user_sub_pref = hist_df.groupby('user_id')['prod_sub_typ'].apply(lambda x: Counter(x)).to_dict()
    user_rsk_pref = hist_df.groupby('user_id')['rsk_lvl'].apply(lambda x: Counter(x)).to_dict()

    # 近 3 天的维度偏好（短期意图，权重更高）
    recent_3d = hist_df[hist_df['acs_tm'] >= day_3d]
    user_recent_action_pref = recent_3d.groupby('user_id')['action_typ'].apply(lambda x: Counter(x)).to_dict()
    user_recent_prod_pref = recent_3d.groupby('user_id')['prod_typ'].apply(lambda x: Counter(x)).to_dict()
    user_recent_sub_pref = recent_3d.groupby('user_id')['prod_sub_typ'].apply(lambda x: Counter(x)).to_dict()

    pension_users = set(
        hist_df[hist_df['prod_sub_typ'].str.contains('养老', na=False)]['user_id'].unique()
    )

    feat = {
        'global_top_items': global_top_items,
        'item_pop_score': item_pop_score,
        'item_recent_score': item_recent_score,
        'valid_combos': valid_combos,
        'user_last_rsk': user_last_rsk,
        'user_last_rsk_str': user_last_rsk_str,
        'user_hist_items': user_hist_items,
        'user_unique_recent_items': user_unique_recent_items,
        'user_last_1day_items': user_last_1day_items,
        'user_last_3day_items': user_last_3day_items,
        'user_last_7day_items': user_last_7day_items,
        'user_last_3day_item_cnt': user_last_3day_item_cnt,
        'user_action_pref': user_action_pref,
        'user_prod_pref': user_prod_pref,
        'user_sub_pref': user_sub_pref,
        'user_rsk_pref': user_rsk_pref,
        'user_recent_action_pref': user_recent_action_pref,
        'user_recent_prod_pref': user_recent_prod_pref,
        'user_recent_sub_pref': user_recent_sub_pref,
        'pension_users': pension_users,
    }

    # region agent log
    agent_log("H4", "build_features exit", {
        "tag": tag,
        "global_items": len(global_top_items),
        "users_with_hist": len(user_hist_items),
        "pension_users": len(pension_users)
    }, "notebook:build_features:end")
    # endregion

    return feat


# =========================
# 3. 单用户预测函数
# =========================
def predict_for_user(user, feat, top_global_n=200):
    user_items = feat['user_hist_items'].get(user, [])
    unique_recent_items = feat['user_unique_recent_items'].get(user, [])
    last_1day = feat['user_last_1day_items'].get(user, set())
    last_3day = feat['user_last_3day_items'].get(user, set())
    last_7day = feat['user_last_7day_items'].get(user, set())
    last_3day_cnt = feat['user_last_3day_item_cnt'].get(user, {})

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

    # ---- 候选集构造 ----
    candidates = []
    seen = set()

    # 1. 用户历史去重 item（按时间从近到远）
    for itm in unique_recent_items:
        if itm not in seen:
            candidates.append(itm)
            seen.add(itm)

    # 2. 用户偏好维度笛卡尔积（限制在全局合法组合内）
    #    引入用户「偏好但未真正发生过」的新组合，提升召回
    if action_pref and prod_pref and sub_pref:
        # 短期偏好 + 长期偏好融合后取 top
        def merge_topk(short_pref, long_pref, k):
            merged = Counter()
            for kk, vv in short_pref.items():
                merged[kk] += 2.0 * vv
            for kk, vv in long_pref.items():
                merged[kk] += 1.0 * vv
            return [x for x, _ in merged.most_common(k)]

        top_actions = merge_topk(r_action_pref, action_pref, 3)
        top_prods = merge_topk(r_prod_pref, prod_pref, 3)
        top_subs = merge_topk(r_sub_pref, sub_pref, 5)
        # 用户的 rsk 偏好 + 最近 rsk 强制纳入
        top_rsks_list = [r for r, _ in rsk_pref.most_common(2)]
        if u_rsk_str not in top_rsks_list:
            top_rsks_list.insert(0, u_rsk_str)
        top_rsks = top_rsks_list[:2]

        for a in top_actions:
            for p in top_prods:
                for s in top_subs:
                    for r in top_rsks:
                        itm = f"{a}|{p}|{s}|{r}"
                        if itm in valid_combos and itm not in seen:
                            candidates.append(itm)
                            seen.add(itm)

    # 3. 全局热门（兜底）
    for itm in feat['global_top_items'][:top_global_n]:
        if itm not in seen:
            candidates.append(itm)
            seen.add(itm)

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

    # 预计算位置 map（O(1) 查询）
    pos_map = {itm: i for i, itm in enumerate(unique_recent_items)}

    scored = []

    for itm in candidates:
        if not isinstance(itm, str) or itm.count('|') != 3:
            continue

        action_typ, prod_typ, prod_sub_typ, rsk_lvl = itm.split('|')
        itm_rsk = rsk_map.get(rsk_lvl, 1)

        # 用户历史频次（长期）
        user_freq_score = item_freq.get(itm, 0) / max_user_freq
        # 全局信号
        global_score = feat['item_pop_score'].get(itm, 0)
        recent_score = feat['item_recent_score'].get(itm, 0)
        # 长期维度偏好
        action_score = action_pref.get(action_typ, 0) / max_action
        prod_score = prod_pref.get(prod_typ, 0) / max_prod
        sub_score = sub_pref.get(prod_sub_typ, 0) / max_sub
        rsk_score = rsk_pref.get(rsk_lvl, 0) / max_rsk
        # 短期(近 3 天) 维度偏好（更强意图）
        r_action_score = r_action_pref.get(action_typ, 0) / max_r_action
        r_prod_score = r_prod_pref.get(prod_typ, 0) / max_r_prod
        r_sub_score = r_sub_pref.get(prod_sub_typ, 0) / max_r_sub
        # 近 3 天该 item 的命中次数（强复发信号）
        win3_score = last_3day_cnt.get(itm, 0) / max_3d_cnt

        score = 0.0
        # 用户历史 / 近窗复发主导：高位 NDCG 收益
        score += 4.0 * user_freq_score
        score += 5.0 * win3_score
        # 全局信号（兜底 + 趋势）
        score += 1.0 * global_score
        score += 2.0 * recent_score
        # 长期维度
        score += 1.2 * action_score
        score += 1.5 * prod_score
        score += 2.2 * sub_score
        score += 0.8 * rsk_score
        # 短期维度（强化对 4/1 的指向性）
        score += 1.5 * r_action_score
        score += 1.8 * r_prod_score
        score += 2.5 * r_sub_score

        # 最近时间窗口出现过（叠加加分）
        if itm in last_1day:
            score += 6.0
        elif itm in last_3day:
            score += 2.5
        elif itm in last_7day:
            score += 1.0

        # 用户最后 N 个不同 item 位置加成
        pos = pos_map.get(itm)
        if pos is not None:
            if pos < 3:
                score += 2.5
            elif pos < 8:
                score += 1.2
            elif pos < 15:
                score += 0.4

        # 风评匹配（用户属性稳定）：相同强加权，不同则强惩罚
        if itm_rsk == u_rsk:
            score *= 1.30
        else:
            rsk_diff = abs(itm_rsk - u_rsk)
            score *= max(0.55, 1.0 - 0.18 * rsk_diff)

        # 养老倾向用户：温和加成（避免过推）
        if is_pen_fan and '养老' in prod_sub_typ:
            score *= 1.35

        scored.append((itm, score))

    preds = [x[0] for x in sorted(scored, key=lambda x: x[1], reverse=True)]

    # 兜底补齐到 20 个
    seen2 = set(preds)
    for itm in feat['global_top_items']:
        if len(preds) >= 20:
            break
        if itm not in seen2 and isinstance(itm, str) and itm.count('|') == 3:
            preds.append(itm)
            seen2.add(itm)

    return preds[:20]


# =========================
# 4. 本地验证：用 3/1-3/31 预测 4/1
# =========================
print("📊 正在执行本地验证：用3月历史预测4月1日...")

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

for n, user in enumerate(eval_users, start=1):
    if n % 5000 == 0:
        print(f"已验证 {n}/{len(eval_users)} 个用户...")

    true_items = val_truth[user]
    preds = predict_for_user(user, feat_val)

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, p in enumerate(preds)
        if p in true_items
    )

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(true_items), 20))
    )

    scores.append(dcg / idcg if idcg > 0 else 0)

score_mean = float(np.mean(scores)) if scores else 0.0

# region agent log
agent_log("H1-H4", "validation finished", {
    "eval_users": len(eval_users),
    "score_mean": score_mean,
    "score_count": len(scores)
}, "notebook:validation_end")
# endregion

print("-" * 30)
print(f"🎯 本地模拟得分: {score_mean:.4f}")
print("-" * 30)


# =========================
# 5. 生成最终提交
# =========================
print("📝 正在生成最终提交文件...")

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
    "max_time": str(test_df['acs_tm'].max()),
    "item_nulls": int(test_df['item'].isna().sum()),
    "bad_item_format": int((test_df['item'].str.count('\\|') != 3).sum())
}, "notebook:load_test")
# endregion

# 最终预测：测试用户历史 + 训练集历史共同构建热门候选
final_hist = pd.concat([
    train_df[train_df['acs_tm'] < '2025-04-01'],
    test_df
], ignore_index=True)

feat_final = build_features(final_hist, tag='final')

rows = []
test_users = test_df['user_id'].drop_duplicates().tolist()

for n, user in enumerate(test_users, start=1):
    if n % 5000 == 0:
        print(f"已生成 {n}/{len(test_users)} 个用户...")

    preds = predict_for_user(user, feat_final)

    if len(preds) != 20:
        # region agent log
        agent_log("H5", "prediction length not 20 before output", {
            "user": user,
            "pred_len": len(preds),
            "preds": preds
        }, "notebook:submit_loop")
        # endregion

    for idx, itm in enumerate(preds, start=1):
        action_typ, prod_typ, prod_sub_typ, rsk_lvl = itm.split('|')
        rows.append([user, action_typ, prod_typ, prod_sub_typ, rsk_lvl, idx])

sub = pd.DataFrame(rows, columns=[
    'user_id', 'action_typ', 'prod_typ', 'prod_sub_typ', 'rsk_lvl', 'index'
])

cnt = sub.groupby('user_id').size()

# region agent log
agent_log("H5", "submission built", {
    "rows": len(sub),
    "users": sub['user_id'].nunique(),
    "expected_users": len(test_users),
    "cnt_min": int(cnt.min()) if len(cnt) else None,
    "cnt_max": int(cnt.max()) if len(cnt) else None,
    "cnt_mean": float(cnt.mean()) if len(cnt) else None,
    "user_set_match": bool(set(sub['user_id']) == set(test_df['user_id']))
}, "notebook:submission_check")
# endregion

assert cnt.min() == 20 and cnt.max() == 20, cnt.describe()
assert set(sub['user_id']) == set(test_df['user_id']), "提交用户与测试集用户不一致"

os.makedirs('/work/task1/submit', exist_ok=True)
sub.to_csv('/work/task1/submit/prediction.csv', index=False, encoding='utf-8-sig')

print("✅ prediction.csv 已生成：/work/task1/submit/prediction.csv")
print(sub.head(25))
print(sub.groupby('user_id').size().describe())
print(f"🪵 Debug日志路径: {DEBUG_LOG_PATH}")