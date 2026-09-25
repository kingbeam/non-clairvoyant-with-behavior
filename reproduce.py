"""
Reproduce the max-stretch (1|r_j, pmtn|S_max) evaluation from the
Prediction+Scheduling notebook.

Usage:
    python reproduce.py   # runs the clip(0.1,10) and no-clip variants

Notes:
    - Extracts cell 4 (the prediction pipeline: CQR/HRAS/Iso/Meta/TwoSt/Rec) and
      cell 8 (max-stretch evaluation) directly from the notebook so the code stays
      identical to the notebook rather than hand-copied.
    - Data is read from the already-extracted per-job CSV files in the local data
      directory.
    - Metric: S_max = raw max stretch; rho_max = S_max/S_emp, rho_99 = p99/S_emp,
      rho_med = median/S_emp, where S_emp is the realized max stretch of EDF at S*,
      used as the OPT anchor.
    - clip_ratio=(0.1,10) is the notebook's shared precondition (clips p_hat with the
      ground truth); clip_ratio=None disables clipping to test sensitivity of
      max-stretch to p_hat tail error.
"""
import json, pathlib, numpy as np

NB = "Prediction+Scheduling.ipynb"
LOCAL_DATA = "data"

nb = json.load(open(NB))


def load_cell(idx):
    return "".join(nb["cells"][idx]["source"])


def build_pipeline():
    """Run cell 4's prediction pipeline, returning (df, idx_te, predictions)."""
    src4 = load_cell(4)
    src4 = src4.replace('pathlib.Path("/content/extracted")', f'pathlib.Path("{LOCAL_DATA}")')
    src4 = src4.replace("/content/extracted", LOCAL_DATA)

    ns4 = {"__name__": "nb4"}
    exec(src4, ns4)
    ns4["extract_archives"] = lambda: None   # data/ already extracted, skip tar extraction

    df, idx_te, predictions, _ = ns4["main"]()
    return df, idx_te, predictions


def build_max_stretch_runner():
    """Run cell 8's max-stretch evaluation, returning run_max_stretch_all_methods."""
    src8 = load_cell(8)
    ns8 = {}
    exec(src8, ns8)
    return ns8["run_max_stretch_all_methods"]


# prediction keys (M1/M3/...) -> real method names in the paper
NAME_MAP = {"M1": "CQR", "M3": "HRAS", "M4": "Iso", "M5": "Meta", "M6": "TwoSt", "M7": "Rec"}


def _fmt_row(name, sprpt, edfp):
    """One row: SPRPT and EDF-P rho_max/rho_99 (two decimals)."""
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

    for clip_ratio, tag in [((0.1, 10.0), "clip(0.1,10)"), (None, "no clip")]:
        res = run(df, idx_te, predictions, sample_size=5000,
                  clip_ratio=clip_ratio, verbose=False)
        _print_results(tag, res)


if __name__ == "__main__":
    main()
