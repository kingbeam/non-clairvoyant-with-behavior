# -*- coding: utf-8 -*-
"""
v2026 spot-gpu 规律图 (regularity / characterization figures)
=============================================================
直接统计 job_info_df.csv,输出数据集的规律图 + 控制台可引用统计。

产出 (figures_v2026/):
  fig_cdf_combined.png     A1+A2  资源(GPU/CPU)与执行时长 CDF(log-x),各面板仅 All(总的)
  fig_hour_of_week_gpu.png B1  周内小时(0-167)GPU(cards)绝对请求 ±95%CI,20h 刻度 + 每日竖线
  fig_hour_of_week_cpu.png B1  周内小时(0-167)CPU(cores)绝对请求 ±95%CI,20h 刻度 + 每日竖线
  fig_weekly.png           B2  相对星期(rel-dow)周提交曲线 ± 跨周 std
  fig_dow_hour_heatmap.png  B3  rel-dow x rel-hour 提交热力图(双周期叠加)
  fig_cardclass_type.png    C1  HP/Spot 卡类构成(左)与 worker 规模构成(右)

说明:
  - submit_time 为相对秒(time 轴是 relative hour/day-of-week;绝对相位未发布)。
  - 时长 CDF 横轴为 log 秒;分位数标注照搬 ATLAS Fig.4 的样式。
用法:
  python plot_2026_regularity.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parent
RAW = BASE / "data_v2026_spot_gpu" / "data" / "job_info_df.csv"
OUT = pathlib.Path("C:/Users/梁祐铭/Desktop/ATLAS/iclr2027/figure")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
C_ALL, C_HP, C_SP = "#555555", "#1f77b4", "#ff7f0e"   # All / HP / Spot

# ---------------------------------------------------------------- load
df = pd.read_csv(RAW)
df["total_gpu"] = df["gpu_request"] * df["worker_num"]   # per-instance 请求 × 实例数 = job 级总量
df["total_cpu"] = df["cpu_request"] * df["worker_num"]
df["p"] = df["duration"].astype(float)
df["day"] = (df["submit_time"] // 86400).astype(int)
df["hour"] = ((df["submit_time"] // 3600) % 24).astype(int)
df["dow"] = (df["day"] % 7).astype(int)

# ---- 相位对齐:相对时间可平移(绝对相位未知),自动把"双日低谷"旋转到显示周的末尾 d5-d6 ----
_wk0 = df.assign(_w=df["day"] // 7)
_rd = _wk0.groupby(["_w", "dow"]).size().unstack(fill_value=0).reindex(columns=range(7), fill_value=0)
_tot0 = _rd.sum(1); _full0 = (_tot0 >= 1000) & (_rd > 0).all(1)
_wd = _rd[_full0].div(_rd[_full0].sum(1), axis=0).mean()
_best, _s0 = None, 0
for _s in range(7):                                  # 找相邻双日平均最低的起始日
    _sc = _wd[_s] + _wd[(_s + 1) % 7]
    if _best is None or _sc < _best:
        _best, _s0 = _sc, _s
ALIGN = (5 - _s0) % 7                               # 使低谷起始日 -> 显示日 5
df["dow"] = ((df["dow"] + ALIGN) % 7).astype(int)    # 重标:低谷现位于 d5-d6(周末在末尾)
df["hour_wk"] = (((df["submit_time"] // 3600) + ALIGN * 24) % 168).astype(int)  # 对齐后的周内小时 0-167

GROUPS = {"All": df, "HP": df[df.job_type == "HP"], "Spot": df[df.job_type == "Spot"]}
GROUP_STYLE = {"All": (C_ALL, "--"), "HP": (C_HP, "-"), "Spot": (C_SP, "-")}

def fmt_dur(s):
    """秒 -> 人可读 ('20 min', '2.1 d', ...),用于图注与统计。"""
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

# ================================================================ A1+A2 combined CDF (仅 All/总的)
qlist = [0.50, 0.75, 0.90, 0.95, 0.99]
qlab = ["50%", "75%", "90%", "95%", "99%"]

def _qfmt(v, col):
    if col == "p":
        return fmt_dur(v)
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"

panels = [("total_gpu", "GPU Units"), ("total_cpu", "CPU Cores"), ("p", "Processing time")]
letters = ("(a)", "(b)", "(c)")
panel_colors = ["#1f77b4", "#2ca02c", "#d62728"]        # GPU=蓝, CPU=绿, Processing time=红
def _draw_cdf(dsub, tag):
    """tag="" 输出未清洗版(fig_cdf_combined.*);tag="_clean" 输出清洗版(fig_cdf_combined_clean.*)。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, xlab), L, pc in zip(axes, panels, letters, panel_colors):
        x = np.sort(dsub[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=pc, lw=1.8)                  # 三条曲线三种颜色,不保留 legend
        ax.text(0.0, 1.05, L, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=12, fontweight="bold")
        # 分位标注:在曲线上的对应点打点;% 标在点上方、对应值标在点下方(上下分布)
        vals = dsub[col].quantile(qlist)
        for lab, q, v in zip(qlab, qlist, vals):
            # 竖直虚线只到曲线交点(数据高 q / ylim 1.07 = 轴内比例),颜色随曲线
            ax.axvline(v, ymin=0.0, ymax=q / 1.07, color=pc, lw=0.6, alpha=0.25, ls=":")
            # 标注整体贴 x 轴:数值在最底、百分比在其上(间距 0.05 拉开两行),加粗
            ax.text(v, 0.010, _qfmt(v, col), ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#333")    # 对应值(最贴 x 轴)
            ax.text(v, 0.010 + 0.05, lab, ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#111")    # 百分比(值上方)
        ax.set_xscale("log")
        ax.set_xlabel(xlab, fontweight="bold")            # x 轴字加粗
        ax.set_ylim(0, 1.07)
        if ax is axes[0]:
            ax.set_ylabel("Cumulative Probability", fontweight="bold")  # y 轴字加粗
        else:
            ax.set_yticklabels([])                        # 其余不保留纵坐标(刻度数字不加粗)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(OUT / f"fig_cdf_combined{tag}.png", bbox_inches="tight", pad_inches=0.0)
    fig.savefig(OUT / f"fig_cdf_combined{tag}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

_draw_cdf(df, "")                                          # 未清洗(全部 466,867 条)
_draw_cdf(df[df["submit_time"] > 0], "_clean")             # 清洗后(463,279 条)

# ================================================================ B1 资源统计(不绘图,供控制台与图1)
df["week"] = (df["day"] // 7).astype(int)                 # 与 B2 共用
# t=0 快照: trace 起始录制时在跑的常驻任务(3,588 个 HP, 时长中位 45 天)被统一打上
# submit_time=0, 这不是真实的提交行为, 且其 duration 记录的是"结束时刻"而非处理时长
# (中位 45 天 vs 其余 20 分钟). 时间规律分析与数据刻画均剔除; CDF 另出清洗版对照。
df_t = df[df["submit_time"] > 0]
_hw = df_t.groupby(["week", "hour_wk"]).size().unstack(fill_value=0).reindex(
    columns=np.arange(168), fill_value=0)
_fullwk = _hw.index[_hw.sum(axis=1) >= 5000]              # 完整周(与提交数口径一致)
_resstat = []                                             # 供末尾控制台打印
for res, lab in [("total_gpu", "GPU (cards)"), ("total_cpu", "CPU (cores)")]:
    agg = df_t.groupby(["week", "hour_wk"])[res].sum().unstack(
        fill_value=0).reindex(columns=np.arange(168), fill_value=0).loc[_fullwk]
    mu = agg.mean()                                       # 各小时跨周平均
    _min = float(mu[mu > 0].min()) if (mu > 0).any() else float(mu.min())
    _resstat.append((lab, float(mu.max() / _min), int(mu.idxmax())))

# B1b submit 提交量统计(不绘图)
_cnt = _hw.loc[_fullwk]                                   # 每完整周每小时的提交数
_mu = _cnt.mean()
_min = float(_mu[_mu > 0].min()) if (_mu > 0).any() else float(_mu.min())
_resstat.append(("Submissions/hour", float(_mu.max() / _min), int(_mu.idxmax())))

# B1c: 计算 GPU | CPU | Submissions 的 mean/CI(不绘图,供图1/图2用)
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

# ================================================================ B2 weekly (rel-dow) 只算统计量(不绘图)
wk = pd.pivot_table(df_t.assign(week=df_t["day"] // 7), index="week", columns="dow",
                    values="job_name", aggfunc="count", fill_value=0)
tot = wk.sum(axis=1)
full = (tot >= 1000) & (wk > 0).all(axis=1)
w = wk[full].div(tot[full], axis=0)
mu_w, sd_w = w.mean(), w.std()
lo = mu_w.nsmallest(2).index.tolist()

# ================================================================ B3 rel-dow x rel-hour heatmap
df_full = df_t[df_t["week"].isin(_fullwk)]   # 与 hour-of-week 图同口径:仅完整周, 剔除 t=0
m = pd.crosstab(df_full["dow"], df_full["hour"]).values  # 真实提交数(非百分比), shape (7, 24)

# Day5-Day6、Day6-Day7 之间各插一个 NaN 行(白色间隙);间隙高 = 行高的 10%
gap = np.full((1, 24), np.nan)
m_gap = np.vstack([m[:5], gap, m[5:6], gap, m[6:]])   # (9, 24)

x_edges = np.arange(25) - 0.5                          # 24 列的边
row_h, gap_h = 1.0, 0.1
y_edges = [0.0]
for i in range(7):
    y_edges.append(y_edges[-1] + row_h)               # 数据行
    if i in (4, 5):
        y_edges.append(y_edges[-1] + gap_h)           # 10% 宽间隙
X, Y = np.meshgrid(x_edges, y_edges)

cmap = plt.cm.Blues.copy()
cmap.set_bad('white')                                 # NaN -> 纯白

# ================================================================ 图1: 资源请求 (GPU + CPU) 横向
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

# ================================================================ 图2: 提交 (热力图 + 提交线) 横向
fig = plt.figure(figsize=(10, 2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.375, 1.515], wspace=0.4)

# 左: 热力图(总量, day x hour)
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

# 右: submissions 线(hour-of-week)
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
print("v2026 spot-gpu key statistics (direct from job_info_df.csv)")
print(f"  jobs={len(df):,}  span={df.submit_time.max()/86400:.1f} d  "
      f"orgs={df.organization.nunique()}  gpu-models={df.gpu_model.nunique()}")
print(f"  HP {len(GROUPS['HP']):,} ({100*len(GROUPS['HP'])/len(df):.1f}%) / "
      f"Spot {len(GROUPS['Spot']):,} ({100*len(GROUPS['Spot'])/len(df):.1f}%)")
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

# ================================================================ 图3: pending 规模分布
# 单机 SRPT 模拟: |P_t| = 已到达未完成 (ready heap) 的 job 数, 与 train_SRPT_v2026 口径一致。
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
# 同色系递进(浅→深)的蓝色梯度,与论文 Blues 配色一致
WCOLORS = {1000: "#c6dbef", 5000: "#9ecae1", 7000: "#4292c6", 10000: "#08519c"}

_pending = {}                                   # window -> 合并后的 |P_t| 样本
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

# 分组柱状图: 横轴=分位(50/75/90/99), 每组 4 根柱 = 各训练窗口的 |P_t| 绝对数
_QLAB = ["50%", "75%", "90%", "99%"]
_QVALS = (50, 75, 90, 99)
_qmat = np.array([[np.percentile(_pending[_nw], _q) for _q in _QVALS]
                  for _nw in WINDOWS])            # (窗口数, 分位数)

# 目标: 以 0.85\textwidth 插入 PDF 时轴标签为 10pt → 字号 = 10 / 缩放比
_FIG_W_IN, _INSERT_FRAC = 10.0, 0.85                 # 收窄画布: 避免图形过宽
_SCALE = _INSERT_FRAC * 6.5 / _FIG_W_IN              # 6.5in = \textwidth
_FS_AX  = 10.0 / _SCALE                              # 轴标签
_FS_TICK = 10.0 / _SCALE                             # 刻度
_FS_NUM  = 7.0 / _SCALE                              # 柱顶数值
_FS_LEG  = 8.5 / _SCALE                              # 图例

fig, ax = plt.subplots(figsize=(_FIG_W_IN, 5.2))       # 纵向稍高: 留出图例位置
_xpos = np.arange(len(_QVALS))
_BARW, _GAP = 0.1394, 0.06                             # 柱宽不变; 组内间隙
_w = _BARW + _GAP                                      # 组内相邻柱的中心距
_fmt = lambda _v: f"{_v:,.0f}" if _v >= 1000 else f"{_v:.0f}"   # 仅千位以上加逗号
for _j, _nw in enumerate(WINDOWS):
    _off = (_j - (len(WINDOWS) - 1) / 2) * _w
    _bars = ax.bar(_xpos + _off, _qmat[_j], _BARW,
                   label=f"$n={_nw:,}$", color=WCOLORS[_nw],
                   edgecolor="#4a4a4a", linewidth=0.5)
    for _b, _v in zip(_bars, _qmat[_j]):
        ax.text(_b.get_x() + _b.get_width() / 2, _v + _qmat.max() * 0.012,
                _fmt(_v), ha="center", va="bottom",
                fontsize=_FS_NUM, color="#1a1a1a")     # 柱顶上方,横排
ax.set_xticks(_xpos)
ax.set_xticklabels(_QLAB)
ax.set_xlabel("Quantile of pending jobs", fontsize=_FS_AX, fontweight="bold")
ax.set_ylabel("Number of pending jobs", fontsize=_FS_AX, fontweight="bold")
ax.tick_params(labelsize=_FS_TICK)
ax.set_ylim(0, _qmat.max() * 1.30)                    # 顶部留白: 让图例与柱顶标签分离
ax.legend(fontsize=_FS_LEG, loc="upper left", frameon=False,
          borderaxespad=0.6, labelspacing=0.32,
          handlelength=1.4, handletextpad=0.6)        # 图内左上角, 纵向排列
fig.tight_layout(pad=0.05)
fig.savefig(OUT / "fig_pending.png", bbox_inches="tight", pad_inches=0.0)
fig.savefig(OUT / "fig_pending.pdf", bbox_inches="tight", pad_inches=0.0)
plt.close(fig)
print("  pending |P_t| quantiles (absolute):")
for _nw, _row in zip(WINDOWS, _qmat):
    print(f"    n={_nw:5d}: "
          + "  ".join(f"p{_q}={_v:,.0f}" for _q, _v in zip(_QVALS, _row)))
