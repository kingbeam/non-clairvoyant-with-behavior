"""
Usage:
    python extract_spot2026_features.py
"""
import pathlib, time, warnings
import numpy as np, pandas as pd, json

warnings.filterwarnings("ignore"); np.random.seed(42)

BASE_DIR = pathlib.Path(__file__).resolve().parent  # .../data_v2026_spot_gpu/
RAW_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jobs():
    df = pd.read_csv(RAW_DIR / "job_info_df.csv")  # has header
    df = df.rename(columns={
        "organization": "user",
        "job_type": "workload",
        "gpu_model": "gpu_type_spec",
    })
    # cast categorical keys to string: integer group keys break pandas MultiIndex reindex
    df["user"] = df["user"].astype(str)
    df = df[df["duration"] > 0].reset_index(drop=True)
    # Drop persistent jobs already running when the trace starts (submit_time=0, 3,588 HP):
    # their duration records an end time rather than processing time, so they are excluded
    # from both the time-pattern analysis and the data (consistent with plot_2026_regularity.py).
    df = df[df["submit_time"] > 0].reset_index(drop=True)
    # per-worker requests → job-level totals (mirrors v2020: plan/100 × inst_num)
    df["total_plan_cpu"] = df["cpu_request"] * df["worker_num"]
    df["total_plan_gpu"] = df["gpu_request"] * df["worker_num"]
    df["total_inst_num"] = df["worker_num"].astype(np.float64)
    df["p_star"] = df["duration"].astype(np.float64)
    df["submit_time"] = df["submit_time"].astype(np.float64)
    return df


def add_basic_features(df):
    df = df.copy()
    for c in ["total_plan_cpu", "total_plan_gpu", "total_inst_num"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    # per-worker CPU is heavy-tailed (0–192) → log1p; GPU (0.01–8) stays raw (bounded)
    df["log_cpu_per_inst"] = np.log1p(df["total_plan_cpu"] / np.maximum(1.0, df["total_inst_num"]))
    df["gpu_per_inst"] = df["total_plan_gpu"] / np.maximum(1.0, df["total_inst_num"])
    # CPU:GPU balance — CPU-bound vs GPU-bound job profile (spot-GPU specific).
    df["log_cpu_per_gpu"] = np.log1p(df["cpu_request"] / np.maximum(df["gpu_request"], 0.01))
    # log_burst: log1p of jobs submitted at the exact same second (batch submission; simultaneous → no leak)
    df["log_burst"] = np.log1p(df.groupby("submit_time")["submit_time"].transform("size"))
    # submit_time is in relative seconds with unknown absolute phase, so dow needs a phase
    # alignment (ALIGN=4): the submission trough (weekend) sits at raw dow=1,2 and shifts
    # to d5-d6 after +4, matching the ALIGN computed in plot_2026_regularity.py.
    hour = ((df["submit_time"] // 3600) % 24).astype(int)
    dow = ((df["submit_time"] // 86400 + 4) % 7).astype(int)
    df["hour"] = hour; df["dow"] = dow
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def make_signatures(df, tr):
    """Signature = user|workload|quantile bins of total cpu/gpu/inst (no mem, no group)."""
    df = df.copy(); tdf = df[tr]

    def qbin(col, q=10):
        e = np.quantile(tdf[col].clip(lower=1e-6), np.linspace(0, 1, q + 1))
        e[0], e[-1] = -np.inf, np.inf
        return np.searchsorted(e, df[col].values, side="right") - 1

    df["cpu_b"] = qbin("total_plan_cpu"); df["gpu_b"] = qbin("total_plan_gpu")
    df["inst_b"] = qbin("total_inst_num")
    df["task_signature"] = (
        df["user"].astype(str) + "|" + df["workload"].astype(str) + "|" +
        df["cpu_b"].astype(str) + "-" + df["gpu_b"].astype(str) + "-" + df["inst_b"].astype(str))
    df["p_star_log"] = np.log1p(df["p_star"])
    global_mu = df.loc[tr, "p_star_log"].mean()
    sig = (df.loc[tr].groupby("task_signature")["p_star_log"]
           .agg(sig_mean="mean", sig_median="median", sig_count="count",
                sig_q25=lambda x: x.quantile(0.25), sig_q75=lambda x: x.quantile(0.75))
           .reset_index())
    n, m = sig["sig_count"].astype(float), sig["sig_mean"].astype(float)
    sig["sig_mean_shrink"] = (n * m + 5.0 * global_mu) / (n + 5.0)
    df = df.merge(sig, on="task_signature", how="left")
    # sig_count is heavy-tailed (0–70k) → log1p for the linear embedding
    df["log_sig_count"] = np.log1p(df["sig_count"].fillna(0))
    return df, global_mu


def add_causal_histories(df, global_mu):
    """Expanding stats per organization (user) — train-only, frozen to val/test."""
    df = df.sort_values("submit_time").reset_index(drop=True)
    df["p_star_log"] = df.get("p_star_log", np.log1p(df["p_star"]))

    t1, t2 = df["submit_time"].quantile([0.70, 0.85]).values
    tr = df["submit_time"] < t1
    va = (df["submit_time"] >= t1) & (df["submit_time"] < t2)
    te = df["submit_time"] >= t2
    df_tr = df[tr].copy()

    key, prefix = "user", "use"
    g = df_tr.groupby(key)["p_star_log"]
    # expanding().groupby().shift() yields a MultiIndex (key, idx); droplevel(0)
    # restores a plain index so the df.loc[] assignment aligns by index value.
    exp_mean = g.expanding(min_periods=1).mean().groupby(key).shift(1).droplevel(0)
    exp_count = g.expanding(min_periods=1).count().groupby(key).shift(1).droplevel(0)
    ewm = g.ewm(span=10, adjust=False).mean().groupby(key).shift(1).droplevel(0)
    # time gap to previous job of the same organization (real seconds, not row gap)
    dt = df_tr.groupby(key)["submit_time"].diff()
    df.loc[tr, f"{prefix}_hist_mean"] = exp_mean.fillna(global_mu)
    df.loc[tr, f"{prefix}_hist_count"] = exp_count.fillna(0).clip(lower=0)
    df.loc[tr, f"{prefix}_ewm"] = ewm.fillna(global_mu)
    df.loc[tr, f"{prefix}_dt_prev"] = np.log1p(dt.fillna(0))

    final_stats = df_tr.groupby(key)["p_star_log"].agg(mean="mean", count="count").to_dict("index")
    for mask in [va, te]:
        df.loc[mask, f"{prefix}_hist_mean"] = df.loc[mask, key].map(
            lambda k: final_stats.get(k, {}).get("mean", global_mu))
        df.loc[mask, f"{prefix}_hist_count"] = df.loc[mask, key].map(
            lambda k: float(final_stats.get(k, {}).get("count", 0)))
        df.loc[mask, f"{prefix}_ewm"] = df.loc[mask, f"{prefix}_hist_mean"]
        df.loc[mask, f"{prefix}_dt_prev"] = 0.0

    # use_hist_count is heavy-tailed (0–157k) → log1p for the linear embedding
    df["log_use_hist_count"] = np.log1p(df[f"{prefix}_hist_count"])

    return df, tr, va, te


def prepare_and_save(df, tr, va, te, global_mu):
    NUM = [
        "log_total_plan_cpu", "log_total_plan_gpu", "log_total_inst_num",
        "log_cpu_per_inst", "gpu_per_inst", "log_cpu_per_gpu",
        "hour", "dow", "sin_hour", "cos_hour", "is_weekend", "log_burst",
        "sig_mean", "sig_median", "sig_q25", "sig_q75", "log_sig_count", "sig_mean_shrink",
        "use_hist_mean", "log_use_hist_count", "use_ewm", "use_dt_prev",
    ]
    NUM = [f for f in NUM if f in df.columns]

    for c in ["sig_mean", "sig_median", "sig_q25", "sig_q75"]:
        if c in df.columns:
            df[c] = df[c].fillna(global_mu)
    df["sig_count"] = df["sig_count"].fillna(0)
    df["sig_mean_shrink"] = df["sig_mean_shrink"].fillna(global_mu)

    cats = {}
    for c in ["user", "workload", "gpu_type_spec"]:
        uvals = df.loc[tr, c].dropna().unique()
        # index 0 is reserved for unseen/unknown (PyTorch nn.Embedding rejects -1);
        # real categories are encoded to 1..N, cardinality = N + 1
        m = {v: i + 1 for i, v in enumerate(uvals)}
        df[f"{c}_enc"] = df[c].map(m).fillna(0).astype(int)
        NUM.append(f"{c}_enc")
        cats[f"{c}_enc"] = len(uvals) + 1

    for name, mask in [("train", tr), ("val", va), ("test", te)]:
        X = df.loc[mask, NUM].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df.loc[mask, "p_star"].values
        out = pd.DataFrame(X.values, columns=NUM)
        out["target_p_star"] = y
        out["target_log1p"] = np.log1p(y)
        out["submit_time"] = df.loc[mask, "submit_time"].values
        out.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {out.shape}")

    with open(OUT_DIR / "feature_config.json", "w") as f:
        json.dump({"n_cont": len(NUM) - len(cats), "n_cat": len(cats),
                   "feats": NUM, "cat_cards": cats}, f, indent=2)
    return NUM, cats


if __name__ == "__main__":
    t0 = time.time()
    print("[1/3] Loading...")
    df = load_jobs()
    print(f"  Jobs: {len(df):,}")

    print("[2/3] Feature engineering...")
    t1, t2 = df["submit_time"].quantile([0.70, 0.85]).values
    tr = df["submit_time"] < t1
    va = (df["submit_time"] >= t1) & (df["submit_time"] < t2)
    te = df["submit_time"] >= t2
    df = add_basic_features(df)
    df, global_mu = make_signatures(df, tr)
    df, tr, va, te = add_causal_histories(df, global_mu)
    print(f"  Train: {tr.sum():,} | Val: {va.sum():,} | Test: {te.sum():,}")

    print("[3/3] Saving...")
    feats, cat_cards = prepare_and_save(df, tr, va, te, global_mu)
    yl = np.log1p(df.loc[tr, "p_star"].values)
    print(f"  Features: {len(feats)} ({len(feats) - len(cat_cards)} numeric + {len(cat_cards)} categorical)")
    print(f"  Cat cards: {cat_cards}")
    print(f"  Target: mean={yl.mean():.4f} std={yl.std():.4f}")
    print(f"  Done in {time.time() - t0:.1f}s")
