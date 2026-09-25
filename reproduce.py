"""
复现 Prediction+Scheduling.ipynb 的 max-stretch（1|r,pmtn|S_max）指标。

用法:
    python reproduce_max_stretch.py            # 跑 clip(0.1,10) 和 无clip 两个版本

说明:
    - 直接从 notebook 提取 cell4（预测 pipeline: CQR/HRAS/Iso/Meta/TwoSt/Rec）和
      cell8（max-stretch 评估）的源码，保证与 notebook 一致，不手抄。
    - 数据用本地已解压的 pai_*.csv（/root/non-clairvoyant-with-predictions-main/data）。
    - 指标: S_max = 原始 max stretch；ρ_max = S_max/S_emp, ρ_99 = p99/S_emp,
      ρ_med = median/S_emp（S_emp = EDF@S* 的 realized max stretch，即 OPT 锚点）。
    - clip_ratio=(0.1,10) 是 notebook 的 "shared precondition"（用真值 clip p̂）；
      clip_ratio=None 则完全不 clip，可对照看 max-stretch 对 p̂ 尾部误差的敏感性。
"""
import json, pathlib, numpy as np

NB = "/root/non-clairvoyant-with-predictions-main/Prediction+Scheduling.ipynb"
LOCAL_DATA = "/root/non-clairvoyant-with-predictions-main/data"

nb = json.load(open(NB))


def load_cell(idx):
    return "".join(nb["cells"][idx]["source"])


def build_pipeline():
    """跑 cell4 的预测 pipeline，返回 (df, idx_te, predictions)。"""
    src4 = load_cell(4)
    src4 = src4.replace('pathlib.Path("/content/extracted")', f'pathlib.Path("{LOCAL_DATA}")')
    src4 = src4.replace("/content/extracted", LOCAL_DATA)

    ns4 = {"__name__": "nb4"}
    exec(src4, ns4)
    ns4["extract_archives"] = lambda: None   # data/ 已解压，跳过 tar 解压

    df, idx_te, predictions, _ = ns4["main"]()
    return df, idx_te, predictions


def build_max_stretch_runner():
    """跑 cell8 的 max-stretch 评估，返回 run_max_stretch_all_methods 函数。"""
    src8 = load_cell(8)
    ns8 = {}
    exec(src8, ns8)
    return ns8["run_max_stretch_all_methods"]


# predictions 的 key（M1/M3/...）→ 论文里的真实方法名
NAME_MAP = {"M1": "CQR", "M3": "HRAS", "M4": "Iso", "M5": "Meta", "M6": "TwoSt", "M7": "Rec"}


def _fmt_row(name, sprpt, edfp):
    """一行：SPRPT 和 EDF-P 的 ρ_max/ρ_99（保留两位小数）。"""
    return (f"{name:8s}  SPRPT ρ_max={sprpt['max_over_OPT']:>9.2f}  ρ_99={sprpt['p99_over_OPT']:>8.2f}   |   "
            f"EDF-P ρ_max={edfp['max_over_OPT']:>9.2f}  ρ_99={edfp['p99_over_OPT']:>8.2f}")


def _print_results(tag, res):
    meta = res["_meta"]
    print("\n" + "=" * 96)
    print(f"MAX-STRETCH  1|r_j,pmtn|S_max   [{tag}]   n={meta['n']:,}   "
          f"S*={meta['S_bisect']:.2f}   S_emp={meta['S_emp']:.2f}")
    print("-" * 96)
    for method_name, m in res["methods"].items():
        print(_fmt_row(NAME_MAP.get(method_name, method_name), m["SPRPT"], m["EDF-P"]))
    print("-" * 96)
    print("BASELINES:")
    print(f"{'Algorithm':22s} {'ρ_max':>9s} {'ρ_99':>8s}")
    for name in ["OPT (EDF at S*)", "SRPT (true)", "FIFO", "LAS/FB"]:
        m = res["baselines"][name]
        print(f"{name:22s} {m['max_over_OPT']:>9.2f} {m['p99_over_OPT']:>8.2f}")
    print("=" * 96)


def main():
    df, idx_te, predictions = build_pipeline()
    run = build_max_stretch_runner()

    for clip_ratio, tag in [((0.1, 10.0), "clip(0.1,10)"), (None, "无 clip")]:
        res = run(df, idx_te, predictions, sample_size=5000,
                  clip_ratio=clip_ratio, verbose=False)
        _print_results(tag, res)


if __name__ == "__main__":
    main()
