import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.workflow_utils import load_yaml


def _resolve_groups(group_cfg: dict, feature_cols: list[str]) -> list[dict]:
    resolved = []
    for group_name, spec in group_cfg.items():
        exact = [str(item) for item in spec.get("exact", [])]
        prefixes = [str(item) for item in spec.get("prefixes", [])]
        members = []
        for feature in feature_cols:
            if feature in exact or any(feature.startswith(prefix) for prefix in prefixes):
                if feature not in members:
                    members.append(feature)
        if members:
            resolved.append(
                {
                    "group": str(group_name),
                    "parent_group": str(spec.get("parent", "")),
                    "members": members,
                }
            )
    return resolved


def _topk_counts(df: pd.DataFrame, value_col: str, top_k: int) -> set[str]:
    if df.empty:
        return set()
    return set(df.sort_values(value_col, ascending=False).head(top_k)["feature"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate feature-level relevance into thesis group summaries.")
    parser.add_argument("--group-config", required=True, help="YAML mapping groups to exact names/prefixes.")
    parser.add_argument("--attention-csv", required=True, help="Attention feature relevance CSV.")
    parser.add_argument("--permutation-csv", required=True, help="Permutation feature relevance CSV.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV for merged group summary.")
    parser.add_argument("--uncovered-csv", default=None, help="Optional CSV listing uncovered features.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k cutoff used for group hit counts.")
    args = parser.parse_args()

    group_cfg = load_yaml(args.group_config).get("groups", {})
    att_df = pd.read_csv(args.attention_csv)
    perm_df = pd.read_csv(args.permutation_csv)

    feature_cols = sorted(set(att_df["feature"].astype(str)) | set(perm_df["feature"].astype(str)))
    groups = _resolve_groups(group_cfg, feature_cols)
    att_top = _topk_counts(att_df, "attention_mean", int(args.top_k))
    perm_top = _topk_counts(perm_df, "importance_mse_norm_mean", int(args.top_k))

    att_map = att_df.set_index("feature")["attention_mean"].to_dict()
    perm_map = perm_df.set_index("feature")["importance_mse_norm_mean"].to_dict()
    total_att = float(att_df["attention_mean"].sum())
    total_perm = float(perm_df["importance_mse_norm_mean"].sum())

    covered = set()
    rows = []
    for group in groups:
        members = group["members"]
        covered.update(members)
        att_values = [float(att_map.get(feature, 0.0)) for feature in members]
        perm_values = [float(perm_map.get(feature, 0.0)) for feature in members]
        rows.append(
            {
                "parent_group": group["parent_group"],
                "group": group["group"],
                "n_features": len(members),
                "attention_sum": float(sum(att_values)),
                "attention_avg": float(sum(att_values) / len(members)),
                "attention_share": float(sum(att_values) / total_att) if total_att else 0.0,
                "attention_top20_count": int(sum(feature in att_top for feature in members)),
                "permutation_sum": float(sum(perm_values)),
                "permutation_avg": float(sum(perm_values) / len(members)),
                "permutation_share": float(sum(perm_values) / total_perm) if total_perm else 0.0,
                "permutation_top20_count": int(sum(feature in perm_top for feature in members)),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["parent_group", "permutation_avg", "attention_avg"],
        ascending=[True, False, False],
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    if args.uncovered_csv:
        uncovered_rows = [
            {"feature": feature}
            for feature in feature_cols
            if feature not in covered
        ]
        pd.DataFrame(uncovered_rows, columns=["feature"]).to_csv(args.uncovered_csv, index=False)

    print(f"Saved grouped feature summary to {output_csv}")


if __name__ == "__main__":
    main()
