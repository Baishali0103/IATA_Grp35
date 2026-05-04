"""
Pulls every metric file in outputs_dir/ into a single master report.

Produces:
    outputs_dir/REPORT_MASTER_SUMMARY.md
    outputs_dir/REPORT_MASTER_SUMMARY.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def flatten_metrics(label, m):
    rows = []
    if "overall" in m:
        rows.append({"approach": label, "split": "overall", **m["overall"]})
    for name, v in m.get("per_dataset", {}).items():
        rows.append({"approach": label, "split": name, **v})
    return rows


def main():
    out_dir = config.OUTPUTS_DIR
    all_rows = []

    for p in sorted(out_dir.glob("metrics_*.json")):
        label = p.stem.replace("metrics_", "")
        with open(p) as f:
            m = json.load(f)
        all_rows.extend(flatten_metrics(label, m))

    if not all_rows:
        print(f"No metrics found in {out_dir}. Run run_evaluation.py first.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "REPORT_MASTER_SUMMARY.csv", index=False)

    with open(out_dir / "REPORT_MASTER_SUMMARY.md", "w") as f:
        f.write("# Master summary of results\n\n")
        f.write(f"Generated from files in `{out_dir}`.\n\n")

        f.write("## Overall (across all test examples)\n\n")
        overall = df[df["split"] == "overall"]
        f.write(overall.drop(columns=["split"]).to_markdown(index=False))
        f.write("\n\n")

        for ds in sorted(df[df["split"] != "overall"]["split"].unique()):
            sub = df[df["split"] == ds]
            f.write(f"## {ds}\n\n")
            f.write(sub.drop(columns=["split"]).to_markdown(index=False))
            f.write("\n\n")

    print(f"Wrote {out_dir / 'REPORT_MASTER_SUMMARY.md'}")
    print(f"Wrote {out_dir / 'REPORT_MASTER_SUMMARY.csv'}")


if __name__ == "__main__":
    main()