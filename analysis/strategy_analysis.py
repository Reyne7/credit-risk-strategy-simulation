"""
信贷审批策略分析（基于公开数据集 Give Me Some Credit）

流程：数据清洗 → 单变量分箱与 IV → 规则挖掘 → 评分卡 → cutoff 选择 → swap set 对比
所有策略效果都在 30% 留出集（test）上评估，规则和评分卡只在 70% 训练集（train）上开发。

用法：
    python analysis/strategy_analysis.py data/cs-training.csv
数据下载：https://www.kaggle.com/c/GiveMeSomeCredit/data （cs-training.csv）
输出：analysis/output/ 下的 JSON 与 CSV，作品集网页直接读取 strategy.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 2026
Y = "bad"
OUT = Path(__file__).resolve().parent / "output"

# ----------------------------------------------------------------------------
# 0. 读取与字段重命名
# ----------------------------------------------------------------------------
RENAME = {
    "SeriousDlqin2yrs": "bad",  # 2 年内出现 90 天以上逾期
    "RevolvingUtilizationOfUnsecuredLines": "util",  # 循环额度使用率
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd30",  # 30–59 天逾期次数
    "DebtRatio": "debt_ratio",  # 负债收入比
    "MonthlyIncome": "income",  # 月收入
    "NumberOfOpenCreditLinesAndLoans": "open_lines",  # 未结清贷款与额度数
    "NumberOfTimes90DaysLate": "dpd90",  # 90 天以上逾期次数
    "NumberRealEstateLoansOrLines": "mortgage",  # 房贷 / 房产抵押额度数
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd60",  # 60–89 天逾期次数
    "NumberOfDependents": "dependents",  # 供养人数
}
LABEL = {
    "util": "循环额度使用率",
    "age": "年龄",
    "dpd30": "30–59 天逾期次数",
    "debt_ratio": "负债收入比",
    "income": "月收入",
    "open_lines": "未结清贷款与额度数",
    "dpd90": "90 天以上逾期次数",
    "mortgage": "房贷 / 抵押额度数",
    "dpd60": "60–89 天逾期次数",
    "dependents": "供养人数",
}


def load(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].rename(columns=RENAME)
    return df


# ----------------------------------------------------------------------------
# 1. 数据清洗：每一条处理都记录原因与影响行数
# ----------------------------------------------------------------------------
def clean(df):
    log = []
    n0 = len(df)
    bad_age = df["age"] < 18
    log.append({"item": "年龄 < 18", "rows": int(bad_age.sum()), "action": "删除（录入错误）"})
    df = df.loc[~bad_age].copy()

    dpd = ["dpd30", "dpd60", "dpd90"]
    code = (df[dpd] >= 96).any(axis=1)
    log.append({
        "item": "逾期次数为 96 / 98",
        "rows": int(code.sum()),
        "bad_rate": round(float(df.loc[code, Y].mean()), 4),
        "action": "不是真实次数而是特殊编码；单独成箱并作为高风险信号，不能当作 96 次参与计算",
    })
    df["dpd_code"] = code.astype(int)
    for c in dpd:
        df.loc[code, c] = np.nan

    util_err = df["util"] > 10
    log.append({
        "item": "额度使用率 > 1000%",
        "rows": int(util_err.sum()),
        "bad_rate": round(float(df.loc[util_err, Y].mean()), 4),
        "action": "违约率反而低于均值，判定为数据错误；单独成箱，不进规则",
    })
    df.loc[util_err, "util"] = np.nan

    inc_na = df["income"].isna()
    log.append({
        "item": "月收入缺失",
        "rows": int(inc_na.sum()),
        "bad_rate": round(float(df.loc[inc_na, Y].mean()), 4),
        "action": "缺失时负债收入比字段存的是负债金额（中位数 1,159），两者都单独成箱",
    })
    df.loc[inc_na, "debt_ratio"] = np.nan

    dep_na = df["dependents"].isna()
    log.append({"item": "供养人数缺失", "rows": int(dep_na.sum()), "action": "单独成箱"})
    return df, log, n0


# ----------------------------------------------------------------------------
# 2. 分箱、WOE、IV（业务可读的手工分箱，缺失 / 异常单独成箱）
# ----------------------------------------------------------------------------
INF = np.inf
BINS = {
    "util": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, INF],
    "age": [18, 30, 40, 50, 60, 70, INF],
    "dpd30": [0, 1, 2, 3, INF],
    "dpd60": [0, 1, 2, INF],
    "dpd90": [0, 1, 2, INF],
    "debt_ratio": [0, 0.3, 0.5, 0.7, 1.0, INF],
    "income": [0, 3000, 5000, 8000, INF],
    "open_lines": [0, 3, 6, 10, 15, INF],
    "mortgage": [0, 1, 2, 3, INF],
    "dependents": [0, 1, 2, 3, INF],
}
MISSING_LABEL = {"util": "异常值", "dpd30": "特殊编码", "dpd60": "特殊编码", "dpd90": "特殊编码", "debt_ratio": "收入缺失", "income": "缺失", "dependents": "缺失"}


def bin_label(edges, i, var):
    lo, hi = edges[i], edges[i + 1]
    pct = var in ("util", "debt_ratio")
    f = (lambda v: f"{v * 100:.0f}%") if pct else (lambda v: f"{v:,.0f}")
    if var in ("dpd30", "dpd60", "dpd90", "mortgage", "dependents"):
        return f"{int(lo)}+" if hi == INF else str(int(lo))
    if hi == INF:
        return f"≥{f(lo)}"
    return f"{f(lo)}–{f(hi)}"


def assign_bins(df, var):
    edges = BINS[var]
    idx = pd.cut(df[var], edges, right=False, labels=False)
    labels = [bin_label(edges, i, var) for i in range(len(edges) - 1)]
    out = idx.map(lambda i: labels[int(i)] if pd.notna(i) else None)
    return out.fillna(MISSING_LABEL.get(var, "缺失")), labels + [MISSING_LABEL.get(var, "缺失")]


def woe_table(df, var):
    b, order = assign_bins(df, var)
    g = df.groupby(b)[Y].agg(n="size", bads="sum").reindex(order).dropna()
    g["goods"] = g["n"] - g["bads"]
    tb, tg = g["bads"].sum(), g["goods"].sum()
    # +0.5 平滑，避免空箱
    pb = (g["bads"] + 0.5) / (tb + 0.5)
    pg = (g["goods"] + 0.5) / (tg + 0.5)
    g["woe"] = np.log(pg / pb)
    g["iv"] = (pg - pb) * g["woe"]
    g["bad_rate"] = g["bads"] / g["n"]
    g["share"] = g["n"] / g["n"].sum()
    return g


def apply_woe(df, var, table):
    b, _ = assign_bins(df, var)
    return b.map(table["woe"]).fillna(0.0)


# ----------------------------------------------------------------------------
# 3. 规则评估
# ----------------------------------------------------------------------------
def rule_metrics(df, hit):
    n, bads = len(df), df[Y].sum()
    h = df.loc[hit]
    return {
        "hit_rate": float(hit.mean()),
        "hit_bad_rate": float(h[Y].mean()) if len(h) else 0.0,
        "lift": float(h[Y].mean() / df[Y].mean()) if len(h) else 0.0,
        "bad_capture": float(h[Y].sum() / bads),
    }


RULES = [
    # id, 名称, 条件（在清洗后的数据上计算）
    ("R01", "历史出现 90 天以上逾期 ≥ 1 次", lambda d: d["dpd90"].fillna(0) >= 1),
    ("R02", "60–89 天逾期 ≥ 1 次", lambda d: d["dpd60"].fillna(0) >= 1),
    ("R03", "逾期次数为特殊编码（96 / 98）", lambda d: d["dpd_code"] == 1),
    ("R04", "30–59 天逾期 ≥ 2 次", lambda d: d["dpd30"].fillna(0) >= 2),
    ("R05", "额度使用率 ≥ 100%", lambda d: d["util"].fillna(0) >= 1.0),
    ("R06", "额度使用率 ≥ 90% 且 年龄 < 40", lambda d: (d["util"].fillna(0) >= 0.9) & (d["age"] < 40)),
    ("R07", "30–59 天逾期 ≥ 1 次 且 额度使用率 ≥ 70%", lambda d: (d["dpd30"].fillna(0) >= 1) & (d["util"].fillna(0) >= 0.7)),
]


# ----------------------------------------------------------------------------
# 4. 评分卡刻度：基准分 600 对应好坏比 15:1，PDO = 20
# ----------------------------------------------------------------------------
BASE, BASE_ODDS, PDO = 600, 15, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE - FACTOR * np.log(BASE_ODDS)


def to_score(p_bad):
    odds_good = (1 - p_bad) / p_bad
    return OFFSET + FACTOR * np.log(odds_good)


def ks(y, s):
    order = np.argsort(s)
    y = np.asarray(y)[order]
    cb = np.cumsum(y) / y.sum()
    cg = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cb - cg)))


# ----------------------------------------------------------------------------
# 5. 收益假设（示例）：用于 cutoff 选择的敏感性分析
# ----------------------------------------------------------------------------
GAIN_GOOD = 1.0  # 每通过一个好客户的收益（单位化）
LOSS_BAD = [5, 10, 15]  # 每通过一个坏客户的损失，是好客户收益的倍数；取 10 为基准


def main(path):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load(path)
    df, clean_log, n_raw = clean(raw)
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df[Y])
    base_bad = float(test[Y].mean())

    # ---- 单变量分析与 IV（train） ----
    tables, ivs = {}, []
    for var in BINS:
        t = woe_table(train, var)
        tables[var] = t
        ivs.append({"var": var, "label": LABEL[var], "iv": float(t["iv"].sum())})
    ivs.sort(key=lambda x: -x["iv"])
    univariate = {
        v: [
            {"bin": k, "n": int(r.n), "share": round(float(r.share), 4), "bad_rate": round(float(r.bad_rate), 4), "woe": round(float(r.woe), 3)}
            for k, r in tables[v].iterrows()
        ]
        for v in BINS
    }

    # ---- 规则：train 上开发，test 上验证；按顺序计算增量贡献 ----
    rules_out = []
    hit_tr_any = pd.Series(False, index=train.index)
    hit_te_any = pd.Series(False, index=test.index)
    for rid, name, fn in RULES:
        h_tr, h_te = fn(train), fn(test)
        m_tr, m_te = rule_metrics(train, h_tr), rule_metrics(test, h_te)
        inc = h_te & ~hit_te_any  # 前面规则都没拒、这条新拒的
        rules_out.append({
            "id": rid, "name": name,
            "train": {k: round(v, 4) for k, v in m_tr.items()},
            "test": {k: round(v, 4) for k, v in m_te.items()},
            "incremental": {
                "hit_rate": round(float(inc.mean()), 4),
                "bad_rate": round(float(test.loc[inc, Y].mean()), 4) if inc.any() else 0.0,
            },
        })
        hit_tr_any |= h_tr
        hit_te_any |= h_te

    # 选定的硬规则：逾期类信号直接拒绝（命中坏率高、可解释、无需模型）
    HARD = ["R01", "R02", "R03"]
    hard_fn = lambda d: np.logical_or.reduce([fn(d) for rid, _, fn in RULES if rid in HARD])
    hard_te = hard_fn(test)

    # ---- 评分卡：WOE + 逻辑回归（train），刻度换算成分数 ----
    feats = [v["var"] for v in ivs if v["iv"] >= 0.02]
    Xtr = pd.DataFrame({v: apply_woe(train, v, tables[v]) for v in feats})
    Xte = pd.DataFrame({v: apply_woe(test, v, tables[v]) for v in feats})
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Xtr, train[Y])
    p_tr, p_te = lr.predict_proba(Xtr)[:, 1], lr.predict_proba(Xte)[:, 1]
    s_tr, s_te = to_score(p_tr), to_score(p_te)
    model = {
        "features": [{"var": v, "label": LABEL[v], "coef": round(float(c), 3)} for v, c in zip(feats, lr.coef_[0])],
        "auc_train": round(float(roc_auc_score(train[Y], p_tr)), 4),
        "auc_test": round(float(roc_auc_score(test[Y], p_te)), 4),
        "ks_train": round(ks(train[Y], s_tr), 4),
        "ks_test": round(ks(test[Y], s_te), 4),
        "scale": {"base": BASE, "base_odds": BASE_ODDS, "pdo": PDO},
    }
    # 分数稳定性：train vs test 的 PSI（同一时期随机切分，只能说明没有过拟合，不能代替跨期监控）
    edges = np.quantile(s_tr, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(s_tr, edges)[0] / len(s_tr)
    b = np.histogram(s_te, edges)[0] / len(s_te)
    model["psi_train_test"] = round(float(np.sum((b - a) * np.log((b + 1e-6) / (a + 1e-6)))), 4)

    test = test.assign(score=s_te, hard=hard_te.astype(int))

    # ---- 分数段表（test） ----
    sb = pd.cut(test["score"], [-np.inf, 520, 540, 560, 580, 600, 620, 640, 660, np.inf], right=False)
    band = test.groupby(sb, observed=True)[Y].agg(n="size", bads="sum").reset_index()
    band["bad_rate"] = band["bads"] / band["n"]
    bands = [{"band": str(r.score).replace("[-inf, ", "<").replace(", inf)", "+").replace("[", "").replace(")", "").replace(", ", "–"), "n": int(r.n), "bad_rate": round(float(r.bad_rate), 4)} for r in band.itertuples()]

    # ---- cutoff 曲线：硬规则 + 分数 ≥ cutoff 才通过 ----
    curve = []
    for c in range(480, 701, 5):
        ok = (test["hard"] == 0) & (test["score"] >= c)
        appr = float(ok.mean())
        br = float(test.loc[ok, Y].mean()) if ok.any() else 0.0
        goods, bads = int((ok & (test[Y] == 0)).sum()), int((ok & (test[Y] == 1)).sum())
        curve.append({
            "cutoff": c, "approval": round(appr, 4), "bad_rate": round(br, 4),
            "bad_capture": round(float(test.loc[~ok, Y].sum() / test[Y].sum()), 4),
            "profit": {str(L): round((goods * GAIN_GOOD - bads * L) / len(test), 4) for L in LOSS_BAD},
        })
    best = {str(L): max(curve, key=lambda r: r["profit"][str(L)])["cutoff"] for L in LOSS_BAD}

    # ---- swap set：原规则策略（R01–R07 全拒）vs 新策略（硬规则 + 分数 cutoff），通过率对齐 ----
    old_ok = ~hit_te_any.reindex(test.index)
    old_appr = float(old_ok.mean())
    # 找到通过率最接近原策略的 cutoff
    cand = [(abs(float(((test["hard"] == 0) & (test["score"] >= c)).mean()) - old_appr), c) for c in np.arange(480, 700, 0.5)]
    c_match = min(cand)[1]
    new_ok = (test["hard"] == 0) & (test["score"] >= c_match)
    cell = lambda m: {"n": int(m.sum()), "share": round(float(m.mean()), 4), "bad_rate": round(float(test.loc[m, Y].mean()), 4) if m.any() else 0.0}
    swap = {
        "cutoff": float(c_match),
        "old": {"name": "原规则策略（R01–R07 命中即拒）", "approval": round(old_appr, 4), "bad_rate": round(float(test.loc[old_ok, Y].mean()), 4)},
        "new": {"name": f"硬规则 R01–R03 + 评分 ≥ {c_match:g}", "approval": round(float(new_ok.mean()), 4), "bad_rate": round(float(test.loc[new_ok, Y].mean()), 4)},
        "both_approve": cell(old_ok & new_ok),
        "swap_in": cell(~old_ok & new_ok),  # 原来拒、现在放
        "swap_out": cell(old_ok & ~new_ok),  # 原来放、现在拒
        "both_reject": cell(~old_ok & ~new_ok),
    }

    # ---- 网页模拟器用的聚合表：分数（5 分一档）× 各规则命中组合 → 件数、坏件数 ----
    test["sbin"] = (np.floor(test["score"] / 5) * 5).clip(470, 720).astype(int)
    rule_bits = np.zeros(len(test), dtype=int)
    for i, (rid, _, fn) in enumerate(RULES):
        rule_bits |= fn(test).to_numpy().astype(int) << i
    test["rbits"] = rule_bits
    grid = test.groupby(["sbin", "rbits"])[Y].agg(n="size", b="sum").reset_index()

    result = {
        "dataset": {
            "name": "Give Me Some Credit（Kaggle 公开数据集）",
            "rows_raw": n_raw, "rows_clean": len(df), "train": len(train), "test": len(test),
            "label": "2 年内出现 90 天以上逾期",
            "base_bad_rate": round(base_bad, 4),
        },
        "cleaning": clean_log,
        "iv": [{**v, "iv": round(v["iv"], 4)} for v in ivs],
        "univariate": univariate,
        "labels": LABEL,
        "rules": rules_out,
        "hard_rules": HARD,
        "model": model,
        "bands": bands,
        "curve": curve,
        "economics": {"gain_good": GAIN_GOOD, "loss_bad": LOSS_BAD, "best_cutoff": best},
        "swap": swap,
        "grid": {"rules": [{"id": r[0], "name": r[1]} for r in RULES], "rows": grid.to_numpy().tolist()},
    }
    (OUT / "strategy.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    pd.DataFrame(ivs).to_csv(OUT / "iv.csv", index=False)
    pd.DataFrame([{"id": r["id"], "name": r["name"], **{f"test_{k}": v for k, v in r["test"].items()}} for r in rules_out]).to_csv(OUT / "rules.csv", index=False)
    pd.DataFrame(curve).drop(columns="profit").to_csv(OUT / "cutoff_curve.csv", index=False)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else "data/cs-training.csv")
    print(json.dumps({k: r[k] for k in ["dataset", "model", "economics", "swap"]}, ensure_ascii=False, indent=1))
    print("IV:", [(v["label"], v["iv"]) for v in r["iv"]])
    for x in r["rules"]:
        print(x["id"], x["name"], "test", x["test"], "inc", x["incremental"])
    print("bands", r["bands"])
