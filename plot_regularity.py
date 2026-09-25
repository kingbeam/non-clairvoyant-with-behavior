# -*- coding: utf-8 -*-
"""
Regularity / characterization figures for the GPU job trace.

Generates CDF, resource-request, submission-pattern, and pending-size figures
from job_info_df.csv, plus a console summary of the key statistics used
throughout the paper.

Usage:
    python plot_regularity.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parent
RAW = BASE / "data_gpu_trace" / "data" / "job_info_df.csv"
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
C_ALL, C_OD, C_PRE = "#555555", "#1f77b4", "#ff7f0e"   # All / OnDemand / Preemptible

# ---------------------------------------------------------------- load
df = pd.read_csv(RAW)
df["total_gpu"] = df["gpu_request"] * df["worker_num"]   # per-worker request x worker count = job-level total
df["total_cpu"] = df["cpu_request"] * df["worker_num"]
df["p"] = df["duration"].astype(float)
df["day"] = (df["submit_time"] // 86400).astype(int)
df["hour"] = ((df["submit_time"] // 3600) % 24).astype(int)
df["dow"] = (df["day"] % 7).astype(int)

# ---- Phase alignment: relative time is shift-invariant (absolute phase unknown);
#      automatically rotate the two-day trough to the end of the display week (d5-d6) ----
_wk0 = df.assign(_w=df["day"] // 7)
_rd = _wk0.groupby(["_w", "dow"]).size().unstack(fill_value=0).reindex(columns=range(7), fill_value=0)
_tot0 = _rd.sum(1); _full0 = (_tot0 >= 1000) & (_rd > 0).all(1)
_wd = _rd[_full0].div(_rd[_full0].sum(1), axis=0).mean()
_best, _s0 = None, 0
for _s in range(7):                                  # find the start day whose adjacent two-day average is lowest
    _sc = _wd[_s] + _wd[(_s + 1) % 7]
    if _best is None or _sc < _best:
        _best, _s0 = _sc, _s
ALIGN = (5 - _s0) % 7                               # shift trough start day to display day 5
df["dow"] = ((df["dow"] + ALIGN) % 7).astype(int)    # relabel: trough now at d5-d6 (weekend at the end)
df["hour_wk"] = (((df["submit_time"] // 3600) + ALIGN * 24) % 168).astype(int)  # aligned hour-of-week 0-167

GROUPS = {"All": df, "OnDemand": df[df.job_type == "OnDemand"], "Preemptible": df[df.job_type == "Preemptible"]}
GROUP_STYLE = {"All": (C_ALL, "--"), "OnDemand": (C_OD, "-"), "Preemptible": (C_PRE, "-")}

def fmt_dur(s):
    """Seconds -> human-readable ('20 min', '2.1 d', ...) for annotations and stats."""
    if s < 90:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s/60:.0f} min"
    if s < 86400:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} d"

def add_cdf(ax, group, color, ls, label):
    x = np.sort(group["p"].values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=color, ls=ls, lw=1.8, label=f"{label} (n={len(x):,})")

# ================================================================ A1+A2 combined CDF (All only)
qlist = [0.50, 0.75, 0.90, 0.95, 0.99]
qlab = ["50%", "75%", "90%", "95%", "99%"]

def _qfmt(v, col):
    if col == "p":
        return fmt_dur(v)
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"

panels = [("total_gpu", "GPU Units"), ("total_cpu", "CPU Cores"), ("p", "Processing time")]
letters = ("(a)", "(b)", "(c)")
panel_colors = ["#1f77b4", "#2ca02c", "#d62728"]        # GPU=blue, CPU=green, processing time=red
def _draw_cdf(dsub, tag):
    """tag="" -> raw (fig_cdf_combined.*); tag="_clean" -> cleaned (fig_cdf_combined_clean.*)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, xlab), L, pc in zip(axes, panels, letters, panel_colors):
        x = np.sort(dsub[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=pc, lw=1.8)                  # three curves, three colors, no legend
        ax.text(0.0, 1.05, L, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=12, fontweight="bold")
        # quantile annotations: mark the point on the curve; % above the point, value below
        vals = dsub[col].quantile(qlist)
        for lab, q, v in zip(qlab, qlist, vals):
            # vertical dotted line only to the curve (data height q / ylim 1.07 = in-axis fraction)
            ax.axvline(v, ymin=0.0, ymax=q / 1.07, color=pc, lw=0.6, alpha=0.25, ls=":")
            # annotation near the x axis: value at the bottom, % above (0.05 spacing), bold
            ax.text(v, 0.010, _qfmt(v, col), ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#333")    # value (closest to x axis)
            ax.text(v, 0.010 + 0.05, lab, ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#111")    # percent (above value)
        ax.set_xscale("log")
        ax.set_xlabel(xlab, fontweight="bold")            # bold x label
        ax.set_ylim(0, 1.07)
        if ax is axes[0]:
            ax.set_ylabel("Cumulative Probability", fontweight="bold")  # bold y label
        else:
            ax.set_yticklabels([])                        # others: no y tick labels
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(OUT / f"fig_cdf_combined{tag}.png", bbox_inches="tight", pad_inches=0.0)
    fig.savefig(OUT / f"fig_cdf_combined{tag}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

_draw_cdf(df, "")                                          # raw (all 466,867 rows)
_draw_cdf(df[df["submit_time"] > 0], "_clean")             # cleaned (463,279 rows)

# ================================================================ B1 resource stats (no plot; for console and Fig 1)
df["week"] = (df["day"] // 7).astype(int)                 # shared with B2
# t=0 snapshot: persistent jobs already running when the trace started (3,588 on-demand,
# median duration 45 days) are all stamped submit_time=0. This is not a real submission
# pattern, and their duration records an end time rather than processing time (median 45 d
# vs 20 min for the rest). They are excluded from both the time-pattern analysis and the
# characterization; the CDF also emits a cleaned version for comparison.
df_t = df[df["submit_time"] > 0]
_hw = df_t.groupby(["week", "hour_wk"]).size().unstack(fill_value=0).reindex(
    columns=np.arange(168), fill_value=0)
_fullwk = _hw.index[_hw.sum(axis=1) >= 5000]              # full weeks (same cutoff as submission counts)
_resstat = []                                             # for the console printout at the end
for res, lab in [("total_gpu", "GPU (cards)"), ("total_cpu", "CPU (cores)")]:
    agg = df_t.groupby(["week", "hour_wk"])[res].sum().unstack(
        fill_value=0).reindex(columns=np.arange(168), fill_value=0).loc[_fullwk]
    mu = agg.mean()                                       # per-hour mean across weeks
    _min = float(mu[mu > 0].min()) if (mu > 0).any() else float(mu.min())
    _resstat.append((lab, float(mu.max() / _min), int(mu.idxmax())))

# B1b submission volume stats (no plot)
_cnt = _hw.loc[_fullwk]                                   # submissions per hour per full week
_mu = _cnt.mean()
_min = float(_mu[_mu > 0].min()) if (_mu > 0).any() else float(_mu.min())
_resstat.append(("Submissions/hour", float(_mu.max() / _min), int(_mu.idxmax())))

# B1c: mean/CI for GPU | CPU | Submissions (no plot; for Fig 1/2)
def _mean_ci(mat):
    m = mat.mean(); n = len(mat)
    se = mat.std(ddof=1) / np.sqrt(n)
    try:
        from scipy import stats as _st
        t = float(_st.t.ppf(0.975, n - 1))
    except Exception:
        t = 2.0
    return m, t * se, n

_tri = []
for res, col, lab, ylab in [
        ("total_gpu", "#1f77b4", "GPU (cards)", "GPU requests"),
        ("total_cpu", "#2ca02c", "CPU (cores)", "CPU requests"),
        (None, "#d62728", "Submissions", "Submissions")]:
    if res is None:
        mat = _hw.loc[_fullwk]
    else:
        mat = df_t.groupby(["week", "hour_wk"])[res].sum().unstack(
            fill_value=0).reindex(columns=np.arange(168), fill_value=0).loc[_fullwk]
    m, ci, n = _mean_ci(mat)
    _tri.append((col, lab, ylab, m, ci, n))
xx = np.arange(168)

# ================================================================ B2 weekly (rel-dow) statistics only (no plot)
wk = pd.pivot_table(df_t.assign(week=df_t["day"] // 7), index="week", columns="dow",
                    values="job_name", aggfunc="count", fill_value=0)
tot = wk.sum(axis=1)
full = (tot >= 1000) & (wk > 0).all(axis=1)
w = wk[full].div(tot[full], axis=0)
mu_w, sd_w = w.mean(), w.std()
lo = mu_w.nsmallest(2).index.tolist()

# ================================================================ B3 rel-dow x rel-hour heatmap
df_full = df_t[df_t["week"].isin(_fullwk)]   # same cutoff as the hour-of-week figure: full weeks only, drop t=0
m = pd.crosstab(df_full["dow"], df_full["hour"]).values  # raw submission counts (not percentages), shape (7, 24)

# insert a NaN row (white gap) between Day5-Day6 and Day6-Day7; gap height = 10% of a row
gap = np.full((1, 24), np.nan)
m_gap = np.vstack([m[:5], gap, m[5:6], gap, m[6:]])   # (9, 24)

x_edges = np.arange(25) - 0.5                          # edges for 24 columns
row_h, gap_h = 1.0, 0.1
y_edges = [0.0]
for i in range(7):
    y_edges.append(y_edges[-1] + row_h)               # data row
    if i in (4, 5):
        y_edges.append(y_edges[-1] + gap_h)           # 10% gap
X, Y = np.meshgrid(x_edges, y_edges)

cmap = plt.cm.Blues.copy()
cmap.set_bad('white')                                 # NaN -> pure white

# ================================================================ Fig 1: resource requests (GPU + CPU), horizontal
fig, axes = plt.subplots(1, 2, figsize=(11, 2.6))
for ax, (col, lab, ylab, m3, ci, n), L in zip(
        axes, _tri[:2], ("(a)", "(b)")):
    ax.plot(xx, m3.values, color=col, lw=1.7)
    ax.fill_between(xx, np.maximum((m3 - ci).values, 0.0), (m3 + ci).values, color=col, alpha=0.15)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylab, fontsize=10)
    ax.tick_params(labelsize=10)
    for h in range(0, 169, 24):
        ax.axvline(h, color="#888888", lw=1.0, alpha=0.8)
    ax.set_xlim(0, 168)
    ax.set_xticks(np.arange(0, 169, 24))
    ax.set_xticklabels([str(h) for h in range(0, 169, 24)])
    ax.set_xlabel("Hours from the beginning of the week", fontsize=10)
    ax.text(0.0, 1.05, L, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=12, fontweight="bold")
fig.tight_layout(pad=0.05); fig.savefig(OUT / "fig_resources.png", bbox_inches="tight", pad_inches=0.0)
fig.savefig(OUT / "fig_resources.pdf", bbox_inches="tight", pad_inches=0.0)
plt.close(fig)

# ================================================================ Fig 2: submissions (heatmap + line), horizontal
fig = plt.figure(figsize=(10, 2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.375, 1.515], wspace=0.4)

# left: heatmap (totals, day x hour)
ax_heat = fig.add_subplot(gs[0, 0])
mesh = ax_heat.pcolormesh(X, Y, m_gap, cmap=cmap, shading='flat')
ax_heat.invert_yaxis()
for s in ax_heat.spines.values():
    s.set_visible(False)
ax_heat.tick_params(length=0, labelsize=10)
ax_heat.set_xticks(range(0, 24, 4)); ax_heat.set_xticklabels(range(0, 24, 4))
ax_heat.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.6, 6.7]); ax_heat.set_yticklabels([f"Day {i+1}" for i in range(7)])
ax_heat.set_xlabel("Hour of day", fontsize=10)
ax_heat.set_ylabel("Day of week", fontsize=10)
ax_heat.text(0.0, 1.05, "(a)", transform=ax_heat.transAxes, va="bottom", ha="left",
             fontsize=12, fontweight="bold")
cb = fig.colorbar(mesh, ax=ax_heat, fraction=0.046, pad=0.04)
cb.set_label("Number of submissions", fontsize=10)
cb.locator = plt.MaxNLocator(5); cb.update_ticks()

# right: submissions line (hour-of-week)
ax_line = fig.add_subplot(gs[0, 1])
col, lab, ylab, m3, ci, n = _tri[2]
ax_line.plot(xx, m3.values, color=col, lw=1.7)
ax_line.fill_between(xx, np.maximum((m3 - ci).values, 0.0), (m3 + ci).values, color=col, alpha=0.15)
ax_line.set_ylim(bottom=0)
ax_line.set_ylabel("Submissions per hour", fontsize=10)
ax_line.text(0.0, 1.05, "(b)", transform=ax_line.transAxes, va="bottom", ha="left",
             fontsize=12, fontweight="bold")
ax_line.tick_params(labelsize=10)
for h in range(0, 169, 24):
    ax_line.axvline(h, color="#888888", lw=1.0, alpha=0.8)
ax_line.set_xlim(0, 168)
ax_line.set_xticks(np.arange(0, 169, 24))
ax_line.set_xticklabels([str(h) for h in range(0, 169, 24)])
ax_line.set_xlabel("Hours from the beginning of the week", fontsize=10)

fig.tight_layout(pad=0.05); fig.savefig(OUT / "fig_submissions.png", bbox_inches="tight", pad_inches=0.0)
fig.savefig(OUT / "fig_submissions.pdf", bbox_inches="tight", pad_inches=0.0)
plt.close(fig)

# ================================================================ console: quotable stats
print("=" * 72)
print("GPU-trace key statistics (direct from job_info_df.csv)")
print(f"  jobs={len(df):,}  span={df.submit_time.max()/86400:.1f} d  "
      f"orgs={df.organization.nunique()}  gpu-models={df.gpu_model.nunique()}")
print(f"  OnDemand {len(GROUPS['OnDemand']):,} ({100*len(GROUPS['OnDemand'])/len(df):.1f}%) / "
      f"Preemptible {len(GROUPS['Preemptible']):,} ({100*len(GROUPS['Preemptible'])/len(df):.1f}%)")
for k, g in GROUPS.items():
    d = g["p"]
    print(f"  [{k:4s}] duration median={fmt_dur(d.median()):>9s} "
          f"p90={fmt_dur(d.quantile(.90)):>9s} p99={fmt_dur(d.quantile(.99)):>9s} "
          f"max={fmt_dur(d.max()):>9s} skew={d.skew():6.1f}")
W = df["p"].sum()
print("  total-duration concentration:")
for th, nm in [(7, ">=7d"), (30, ">=30d"), (90, ">=90d"), (150, ">=150d")]:
    m = df["p"] >= th * 86400
    print(f"    jobs {nm:6s}: {m.sum():7,d} ({100*m.mean():.3f}%)  |  "
          f"{100*df[m]['p'].sum()/W:5.1f}% of total duration")
te = df[df.submit_time >= df.submit_time.quantile(.85)]
print(f"  test-split super-long(>=90d) count = {(te['p']>=90*86400).sum()} "
      f"(tail absent in test window)")
for _nm, _amp, _pk in _resstat:
    print(f"  hour-of-week [{_nm}]: peak rel hour {_pk} (d{_pk//24} {_pk%24}:00) | "
          f"amplitude {_amp:.2f}x over {len(_fullwk)} full weeks")
print("  weekly : two-low rel-dows", lo, f" amplitude {mu_w.max()/mu_w.min():.2f}x "
      f"over {len(w)} full weeks")
print("  figures ->", OUT)
print("=" * 72)

# ================================================================ Fig 3: pending size distribution
# Single-machine SRPT simulation: |P_t| = number of arrived-but-unfinished jobs in the ready
# heap, consistent with the SRPT trainer.
import heapq as _hq

def _srpt_pending(arr, times):
    n = len(arr); order = sorted(range(n), key=lambda j: arr[j])
    rem = times.copy().astype(float); ready = []; i = 0
    t = float(arr.min()); sizes = []
    while i < n or ready:
        if not ready and i < n:
            t = max(t, arr[order[i]])
        while i < n and arr[order[i]] <= t + 1e-6:
            j = order[i]; _hq.heappush(ready, (rem[j], j)); i += 1
        if not ready:
            continue
        sizes.append(len(ready))
        r, j = _hq.heappop(ready)
        nxt = arr[order[i]] if i < n else float("inf")
        if r <= nxt - t + 1e-9:
            t += r
        else:
            t = nxt; _hq.heappush(ready, (r - (nxt - t), j))
    return np.array(sizes)

_rng = np.random.RandomState(0)
_raw = pd.read_csv(RAW)
_raw = _raw[_raw["duration"] > 0].sort_values("submit_time").reset_index(drop=True)
WINDOWS, N_REP = [1000, 5000, 7000, 10000], 6
# progressive blue gradient (light -> dark), consistent with the Blues colormap in the paper
WCOLORS = {1000: "#c6dbef", 5000: "#9ecae1", 7000: "#4292c6", 10000: "#08519c"}

_pending = {}                                   # window -> concatenated |P_t| samples
for _nw in WINDOWS:
    _sz = []
    for _k in range(N_REP):
        _s = _rng.randint(0, max(1, len(_raw) - _nw))
        _sub = _raw.iloc[_s:_s + _nw]
        _a = _sub["submit_time"].values.astype(float); _a = _a - _a.min()
        _sz.append(_srpt_pending(_a, _sub["duration"].values.astype(float)))
    _pending[_nw] = np.concatenate(_sz)
    _qq = np.percentile(_pending[_nw], [50, 75, 90, 99])
    print(f"  pending |P_t| (window={_nw:5d}, {N_REP} windows): "
          f"p50={_qq[0]:.0f} p75={_qq[1]:.0f} p90={_qq[2]:.0f} "
          f"p99={_qq[3]:.0f} max={_pending[_nw].max():.0f}")

# grouped bars: x = quantile (50/75/90/99), 4 bars per group = |P_t| counts per training window
_QLAB = ["50%", "75%", "90%", "99%"]
_QVALS = (50, 75, 90, 99)
_qmat = np.array([[np.percentile(_pending[_nw], _q) for _q in _QVALS]
                  for _nw in WINDOWS])            # (windows, quantiles)

# goal: axis labels read 10pt when inserted at 0.85\textwidth in the PDF -> font size = 10 / scale
_FIG_W_IN, _INSERT_FRAC = 10.0, 0.85                 # narrower canvas: avoid over-wide figure
_SCALE = _INSERT_FRAC * 6.5 / _FIG_W_IN              # 6.5in = \textwidth
_FS_AX  = 10.0 / _SCALE                              # axis labels
_FS_TICK = 10.0 / _SCALE                             # ticks
_FS_NUM  = 7.0 / _SCALE                              # bar-top values
_FS_LEG  = 8.5 / _SCALE                              # legend

fig, ax = plt.subplots(figsize=(_FIG_W_IN, 5.2))       # taller: leave room for the legend
_xpos = np.arange(len(_QVALS))
_BARW, _GAP = 0.1394, 0.06                             # fixed bar width; intra-group gap
_w = _BARW + _GAP                                      # center distance between adjacent bars
_fmt = lambda _v: f"{_v:,.0f}" if _v >= 1000 else f"{_v:.0f}"   # comma only for thousands
for _j, _nw in enumerate(WINDOWS):
    _off = (_j - (len(WINDOWS) - 1) / 2) * _w
    _bars = ax.bar(_xpos + _off, _qmat[_j], _BARW,
                   label=f"$n={_nw:,}$", color=WCOLORS[_nw],
                   edgecolor="#4a4a4a", linewidth=0.5)
    for _b, _v in zip(_bars, _qmat[_j]):
        ax.text(_b.get_x() + _b.get_width() / 2, _v + _qmat.max() * 0.012,
                _fmt(_v), ha="center", va="bottom",
                fontsize=_FS_NUM, color="#1a1a1a")     # above bar top, horizontal
ax.set_xticks(_xpos)
ax.set_xticklabels(_QLAB)
ax.set_xlabel("Quantile of pending jobs", fontsize=_FS_AX, fontweight="bold")
ax.set_ylabel("Number of pending jobs", fontsize=_FS_AX, fontweight="bold")
ax.tick_params(labelsize=_FS_TICK)
ax.set_ylim(0, _qmat.max() * 1.30)                    # top whitespace: separate legend from bar-top labels
ax.legend(fontsize=_FS_LEG, loc="upper left", frameon=False,
          borderaxespad=0.6, labelspacing=0.32,
          handlelength=1.4, handletextpad=0.6)        # top-left inside the axes, vertical
fig.tight_layout(pad=0.05)
fig.savefig(OUT / "fig_pending.png", bbox_inches="tight", pad_inches=0.0)
fig.savefig(OUT / "fig_pending.pdf", bbox_inches="tight", pad_inches=0.0)
plt.close(fig)
print("  pending |P_t| quantiles (absolute):")
for _nw, _row in zip(WINDOWS, _qmat):
    print(f"    n={_nw:5d}: "
          + "  ".join(f"p{_q}={_v:,.0f}" for _q, _v in zip(_QVALS, _row)))
