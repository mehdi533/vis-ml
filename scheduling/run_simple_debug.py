from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scheduling.problem import run_optimization
    from scheduling.utils import load_optimization_config
except ModuleNotFoundError:
    from final_optimization_folder.problem import run_optimization
    from final_optimization_folder.utils import load_optimization_config


BLOCK_ORDER = ("ed", "input", "output", "nn", "line", "n1", "n1_redispatch")


def _parse_blocks(raw: str) -> list[str]:
    items = [part.strip().lower() for part in str(raw).split(",")]
    return [item for item in items if item]


def _normalize_blocks(blocks: list[str]) -> list[str]:
    allowed = set(BLOCK_ORDER)
    bad = [name for name in blocks if name not in allowed]
    if bad:
        raise ValueError(f"Unknown block names: {bad}. Allowed: {list(BLOCK_ORDER)}")

    expanded = set(blocks or ["ed"])
    expanded.add("ed")
    if "n1_redispatch" in expanded:
        expanded.update({"n1", "line"})
    if "n1" in expanded:
        expanded.add("line")
    return [name for name in BLOCK_ORDER if name in expanded]


def _make_run_tag(blocks: list[str], explicit_tag: str | None) -> str:
    if explicit_tag:
        return explicit_tag.strip()
    return "__".join(blocks) if blocks else "ed"


def _configure_debug_run(
    cfg: dict,
    *,
    blocks: list[str],
    run_tag: str,
    output_root: Path,
) -> dict:
    out = deepcopy(cfg)
    constraints = dict(out.get("constraints", {}) or {})
    constraints["use_input"] = "input" in blocks
    constraints["use_output"] = "output" in blocks
    constraints["use_nn"] = "nn" in blocks
    constraints["use_line"] = "line" in blocks
    constraints["use_n1"] = "n1" in blocks
    constraints["use_n1_redispatch"] = "n1_redispatch" in blocks
    constraints["use_ed"] = "ed" in blocks
    constraints["nn_mode"] = "milp"
    out["constraints"] = constraints

    solver = dict(out.get("solver", {}) or {})
    solver["feasibility_checks"] = True
    out["solver"] = solver

    plots = dict(out.get("plots", {}) or {})
    plots["enabled"] = False
    out["plots"] = plots

    formulation = dict(out.get("formulation", {}) or {})
    formulation["id"] = run_tag
    formulation["name"] = f"Simple Debug: {', '.join(blocks)}"
    formulation["description"] = f"Minimal debug run with active blocks: {', '.join(blocks)}"
    out["formulation"] = formulation

    output_cfg = dict(out.get("output", {}) or {})
    output_cfg["run_tag"] = run_tag
    output_cfg["results_dir"] = str(output_root)
    output_cfg["log_file"] = str((output_root / f"{run_tag}.log").resolve())
    out["output"] = output_cfg
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one simple optimization with an explicit set of active constraint blocks."
    )
    parser.add_argument(
        "--config",
        default="results/thesis_optimization_results/configs/base_optimization.yaml",
        help="Base optimization config.",
    )
    parser.add_argument(
        "--blocks",
        default="ed",
        help="Comma-separated block list. Example: ed,input,nn,line,n1",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional run tag. Default is derived from the block list.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional output directory. Default is results/thesis_optimization_results/local_validation/simple_debug/<tag>",
    )
    args = parser.parse_args()

    blocks = _normalize_blocks(_parse_blocks(args.blocks))
    run_tag = _make_run_tag(blocks, args.tag or None)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (ROOT / "results" / "thesis_optimization_results" / "local_validation" / "simple_debug" / run_tag).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_optimization_config(args.config)
    debug_cfg = _configure_debug_run(
        cfg,
        blocks=blocks,
        run_tag=run_tag,
        output_root=output_root,
    )

    resolved_cfg_path = output_root / f"{run_tag}_resolved_config.yaml"
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(debug_cfg, f, sort_keys=False)

    res = run_optimization(debug_cfg, config_path=str(resolved_cfg_path))
    print(f"[run_simple_debug] blocks={','.join(blocks)}")
    print(f"[run_simple_debug] status={res['status']} objective={res['objective']}")
    print(f"[run_simple_debug] summary={res['summary_json']}")
    print(f"[run_simple_debug] resolved_config={resolved_cfg_path}")


if __name__ == "__main__":
    main()
