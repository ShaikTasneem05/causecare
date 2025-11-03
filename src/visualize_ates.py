import json
from pathlib import Path
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

def latest_json(path: Path):
    files = sorted(path.glob("causal_results_full_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def load_results(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    parsed = {}
    for est_label, est_info in results.items():
        ate = est_info.get("ate")
        ci = est_info.get("ci")
        p_placebo = None
        refuts = est_info.get("refutations", {})
        if isinstance(refuts, dict):
            place = refuts.get("placebo_treatment_refuter") or refuts.get("placebo")
            if isinstance(place, dict):
                p_placebo = place.get("p_value")
        parsed[est_label] = {
            "ate": ate,
            "ci": ci,
            "placebo_p": p_placebo
        }
    return parsed, data.get("timestamp", "")

def prepare_plot_data(parsed):
    labels = []
    ates = []
    err_low = []
    err_high = []
    pvals = []
    for k, v in parsed.items():
        labels.append(k)
        a = v.get("ate")
        if a is None or (isinstance(a, float) and math.isnan(a)):
            ates.append(0.0)
        else:
            ates.append(float(a))
        ci = v.get("ci")
        if ci and isinstance(ci, (list, tuple)) and len(ci) >= 2:
            low, high = float(ci[0]), float(ci[1])
            err_low.append(max(0.0001, ates[-1] - low))
            err_high.append(max(0.0001, high - ates[-1]))
        else:
            # small dummy error bar so bar is visible
            err_low.append(abs(ates[-1]) * 0.05 + 0.005)
            err_high.append(abs(ates[-1]) * 0.05 + 0.005)
        pvals.append(v.get("placebo_p"))
    return labels, np.array(ates), np.array([err_low, err_high]), pvals

def make_plot(labels, ates, errors, pvals, out_path: Path, title=None):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.6), 5))
    yerr = errors
    ax.bar(x, ates, yerr=yerr, capsize=6)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("ATE (effect on HeartDisease)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("ATE estimates by estimator (with CI / bootstrap if available)")

    for i, (val, p) in enumerate(zip(ates, pvals)):
        ax.text(i, val + (errors[1][i] * 0.6 if errors is not None else 0.01),
                f"{val:.4f}", ha='center', va='bottom', fontsize=9)
        if p is not None:
            ax.text(i, min(0, val) - (errors[0][i] * 1.2),
                    f"placebo p={p:.2f}", ha='center', va='top', fontsize=8, color='dimgray')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def print_console_summary(labels, ates, errors, pvals, ts):
    print("\nATE Visualization summary (generated from timestamp: {})".format(ts))
    for l, a, low, high, p in zip(labels, ates, errors[0], errors[1], pvals):
        low_val = a - low
        high_val = a + high
        print(f"- {l}: ATE={a:.6f}, CI≈[{low_val:.6f}, {high_val:.6f}], placebo_p={p}")

def main():
    args = sys.argv[1:]
    json_path = None
    if args:
        candidate = Path(args[0])
        if candidate.exists():
            json_path = candidate
        else:
            print("Given json path not found:", candidate)
            return
    else:
        src_outputs = Path("outputs")
        latest = latest_json(src_outputs)
        if latest is None:
            print("No causal_results_full_*.json found in outputs/ — please run causal analysis first or provide a JSON path.")
            return
        json_path = latest

    parsed, ts = load_results(json_path)
    labels, ates, errors, pvals = prepare_plot_data(parsed)
    out_png = Path("outputs/ate_comparison.png")
    title = f"ATE Comparison — {json_path.name}"
    saved = make_plot(labels, ates, errors, pvals, out_png, title=title)
    print(f"\nSaved ATE plot to: {saved}")
    print_console_summary(labels, ates, errors, pvals, ts)

if __name__ == "__main__":
    main()
