from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from data_generation.system_adapter import SystemAdapter


@dataclass
class DisturbanceSpec:
    """Structured disturbance plan produced by the dispatcher."""
    kind: str
    line_uid: int = -1
    line_idx: Optional[object] = None
    trip_time: float = float("nan")
    load_step_time: float = float("nan")
    load_step_scale: float = 1.0
    meta: Dict[str, object] = field(default_factory=dict)

    def contingency(self) -> Optional[Dict[str, object]]:
        """Return contingency."""
        raw = self.meta.get("contingency")
        if isinstance(raw, dict):
            return dict(raw)
        if self.line_uid < 0:
            return None
        out: Dict[str, object] = {"uid": int(self.line_uid)}
        if self.line_idx is not None:
            out["idx"] = self.line_idx
        if "bus1" in self.meta:
            out["bus1"] = self.meta["bus1"]
        if "bus2" in self.meta:
            out["bus2"] = self.meta["bus2"]
        return out


class DisturbanceHandler(Protocol):
    """Handler implementation for 'DisturbanceHandler'."""
    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        ...

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        ...


def _sample_from_bins(bins: Sequence[Dict], rng: np.random.Generator) -> Tuple[float, str]:
    """Internal helper to sample from bins."""
    probs = np.asarray([b.get("prob", 1.0) for b in bins], dtype=float)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(bins), p=probs))
    bin_cfg = bins[idx]
    low, high = float(bin_cfg["low"]), float(bin_cfg["high"])
    val = float(rng.uniform(low, high))
    label = str(bin_cfg.get("label", f"bin{idx}"))
    return val, label


def _sample_scalar(
    range_cfg: Dict,
    rng: np.random.Generator,
    *,
    default_low: float,
    default_high: float,
    log_uniform: bool = False,
) -> float:
    """Internal helper to sample scalar."""
    low = float(range_cfg.get("low", default_low))
    high = float(range_cfg.get("high", default_high))
    if log_uniform:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def sample_value(
    value_cfg: Dict,
    rng: np.random.Generator,
    *,
    default_low: float,
    default_high: float,
) -> Tuple[float, str]:
    """Return sample value."""
    if isinstance(value_cfg, (list, tuple)) and len(value_cfg) >= 2:
        value_cfg = {"low": value_cfg[0], "high": value_cfg[1]}
    if isinstance(value_cfg, (int, float)):
        value_cfg = {"low": value_cfg, "high": value_cfg}
    if "bins" in value_cfg and value_cfg["bins"]:
        return _sample_from_bins(value_cfg["bins"], rng)
    log_u = bool(value_cfg.get("log_uniform", False))
    val = _sample_scalar(
        value_cfg,
        rng,
        default_low=default_low,
        default_high=default_high,
        log_uniform=log_u,
    )
    return val, "uniform_log" if log_u else "uniform"


def select_step_targets(
    ss,
    load_cfg: Dict,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Select step targets."""
    adapter = SystemAdapter(ss)
    pq_names_cfg = list(load_cfg.get("pq_names") or [])
    if not pq_names_cfg:
        pq_names_cfg = adapter.pq_names()

    owner_values = [str(o) for o in list(load_cfg.get("owners") or [])]
    if bool(load_cfg.get("random_owner_per_sim", False)) and owner_values:
        if rng is None:
            raise ValueError("random_owner_per_sim requires an RNG.")
        sampled_owner = str(rng.choice(owner_values))
        owner_values = [sampled_owner]

    owner_filter = set(owner_values)
    if owner_filter:
        name_to_owner = adapter.pq_name_to_owner()
        pq_names_cfg = [name for name in pq_names_cfg if name_to_owner.get(str(name)) in owner_filter]

    return pq_names_cfg


def _extract_line_records(ss) -> List[Dict[str, float | int | str]]:
    """Internal helper to extract line records."""
    return SystemAdapter(ss).line_records()


def pick_line_contingencies(
    ss,
    cont_cfg: Dict,
    rng: np.random.Generator,
) -> List[Dict[str, float | int | str]]:
    """Pick line contingencies."""
    line_records = _extract_line_records(ss)
    if not line_records:
        return []

    active = [r for r in line_records if bool(r.get("in_service", True))]
    line_ids_cfg = list(cont_cfg.get("line_ids") or [])
    if line_ids_cfg:
        wanted = {str(v) for v in line_ids_cfg}
        selected = [
            r
            for r in active
            if (str(r["idx"]) in wanted) or (str(r["name"]) in wanted) or (str(r["uid"]) in wanted)
        ]
    else:
        selected = active

    max_lines = int(cont_cfg.get("max_lines", 0) or 0)
    if max_lines > 0 and len(selected) > max_lines:
        pick = rng.choice(len(selected), size=max_lines, replace=False)
        selected = [selected[int(i)] for i in np.sort(pick)]

    return selected


def _resolve_disturbance_configs(cfg: Dict) -> tuple[str, Dict, Dict]:
    """Internal helper to resolve disturbance configs."""
    cont_cfg = cfg.get("contingency", {}) or {}
    load_step_cfg = dict(cont_cfg.get("load_step", {}) or {})
    line_n1_cfg = dict(cont_cfg.get("line_n1", {}) or {})

    if ("mode" in cont_cfg) and (not load_step_cfg) and (not line_n1_cfg):
        cont_mode = str(cont_cfg.get("mode", "none")).lower()
        valid_modes = {"none", "load_step", "line_n1"}
        if cont_mode not in valid_modes:
            raise ValueError(
                f"Unsupported contingency.mode={cont_mode!r}. Expected one of {sorted(valid_modes)}."
            )
        load_step_cfg["enable"] = bool(cont_mode == "load_step" or cont_cfg.get("include_load_step", False))
        line_n1_cfg["enable"] = bool(cont_mode == "line_n1")
        line_n1_cfg["trip_time"] = cont_cfg.get("trip_time")
        line_n1_cfg["line_ids"] = cont_cfg.get("line_ids")
        line_n1_cfg["max_lines"] = cont_cfg.get("max_lines")

    load_step_enabled = bool(load_step_cfg.get("enable", False))
    line_n1_enabled = bool(line_n1_cfg.get("enable", False))

    if load_step_enabled and line_n1_enabled:
        kind = "line_plus_load"
    elif line_n1_enabled:
        kind = "line_n1"
    elif load_step_enabled:
        kind = "load_step"
    else:
        kind = "none"

    return kind, load_step_cfg, line_n1_cfg


def resolve_disturbance_kind(cfg: Dict) -> str:
    """Resolve disturbance kind."""
    kind, _, _ = _resolve_disturbance_configs(cfg)
    return kind


def _load_step_settings(cfg: Dict, load_step_cfg: Dict, rng: np.random.Generator) -> tuple[float, float, str]:
    """Internal helper to load step settings."""
    load_step_time = float(load_step_cfg.get("time", cfg.get("tds", {}).get("load_step_time", 0.1)))
    load_step_scale_cfg = load_step_cfg.get("scale", cfg.get("load_step_scale", {}))
    default_low = float(load_step_scale_cfg.get("low", 1.0))
    default_high = float(load_step_scale_cfg.get("high", default_low))
    step_scale, step_bin = sample_value(
        load_step_scale_cfg,
        rng,
        default_low=default_low,
        default_high=default_high,
    )
    return load_step_time, step_scale, step_bin


def _apply_load_step(ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
    """Internal helper to apply load step."""
    adapter = SystemAdapter(ss)
    step_targets = select_step_targets(ss, cfg.get("load", {}), rng=rng)
    for dev in step_targets:
        adapter.add_load_step(
            dev=dev,
            time_s=float(spec.load_step_time),
            scale=float(spec.load_step_scale),
        )


def _apply_line_toggle(ss, spec: DisturbanceSpec) -> None:
    """Internal helper to apply line toggle."""
    if spec.line_idx is None:
        return
    adapter = SystemAdapter(ss)
    adapter.add_line_toggle(dev=spec.line_idx, time_s=float(spec.trip_time))


class _NoneHandler:
    """Handler implementation for 'NoneHandler'."""
    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        _ = ss, rng
        _, load_step_cfg, _ = _resolve_disturbance_configs(cfg)
        load_step_time = float(load_step_cfg.get("time", cfg.get("tds", {}).get("load_step_time", 0.1)))
        return [
            DisturbanceSpec(
                kind="none",
                load_step_time=load_step_time,
                load_step_scale=1.0,
                meta={"step_bin_label": "disabled"},
            )
        ]

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        _ = ss, spec, cfg, rng


class _LoadStepHandler:
    """Handler implementation for 'LoadStepHandler'."""
    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        _ = ss
        _, load_step_cfg, _ = _resolve_disturbance_configs(cfg)
        load_step_time, step_scale, step_bin = _load_step_settings(cfg, load_step_cfg, rng)
        return [
            DisturbanceSpec(
                kind="load_step",
                load_step_time=load_step_time,
                load_step_scale=step_scale,
                meta={"step_bin_label": step_bin},
            )
        ]

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        _apply_load_step(ss, spec, cfg, rng)


class _LineN1Handler:
    """Handler implementation for 'LineN1Handler'."""
    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        if ss is None:
            raise ValueError("Line-N1 disturbance planning requires a loaded system.")
        _, load_step_cfg, line_n1_cfg = _resolve_disturbance_configs(cfg)
        load_step_time = float(load_step_cfg.get("time", cfg.get("tds", {}).get("load_step_time", 0.1)))
        line_records = pick_line_contingencies(ss, line_n1_cfg, rng)
        if not line_records:
            raise ValueError(
                "contingency.line_n1.enable=true, but no valid in-service lines were found "
                "for contingency.line_n1.line_ids/max_lines."
            )
        trip_time = float(line_n1_cfg.get("trip_time", load_step_time))
        specs: List[DisturbanceSpec] = []
        for record in line_records:
            specs.append(
                DisturbanceSpec(
                    kind="line_n1",
                    line_uid=int(record.get("uid", -1)),
                    line_idx=record.get("idx"),
                    trip_time=trip_time,
                    load_step_time=load_step_time,
                    load_step_scale=1.0,
                    meta={"step_bin_label": "disabled", "contingency": dict(record)},
                )
            )
        return specs

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        _ = cfg, rng
        _apply_line_toggle(ss, spec)


class _LinePlusLoadHandler:
    """Handler implementation for 'LinePlusLoadHandler'."""
    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        if ss is None:
            raise ValueError("Line+load disturbance planning requires a loaded system.")
        _, load_step_cfg, line_n1_cfg = _resolve_disturbance_configs(cfg)
        load_step_time, step_scale, step_bin = _load_step_settings(cfg, load_step_cfg, rng)
        line_records = pick_line_contingencies(ss, line_n1_cfg, rng)
        if not line_records:
            raise ValueError(
                "contingency.line_n1.enable=true, but no valid in-service lines were found "
                "for contingency.line_n1.line_ids/max_lines."
            )
        trip_time = float(line_n1_cfg.get("trip_time", load_step_time))
        specs: List[DisturbanceSpec] = []
        for record in line_records:
            specs.append(
                DisturbanceSpec(
                    kind="line_plus_load",
                    line_uid=int(record.get("uid", -1)),
                    line_idx=record.get("idx"),
                    trip_time=trip_time,
                    load_step_time=load_step_time,
                    load_step_scale=step_scale,
                    meta={"step_bin_label": step_bin, "contingency": dict(record)},
                )
            )
        return specs

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        _apply_load_step(ss, spec, cfg, rng)
        _apply_line_toggle(ss, spec)


HANDLERS: Dict[str, DisturbanceHandler] = {
    "none": _NoneHandler(),
    "load_step": _LoadStepHandler(),
    "line_n1": _LineN1Handler(),
    "line_plus_load": _LinePlusLoadHandler(),
}


class DisturbanceDispatcher:
    """Plan and apply disturbances via handler registry lookup."""
    def __init__(self, handlers: Optional[Dict[str, DisturbanceHandler]] = None) -> None:
        """Initialize this instance."""
        self.handlers = dict(HANDLERS if handlers is None else handlers)

    def plan(self, ss, cfg: Dict, rng: np.random.Generator) -> List[DisturbanceSpec]:
        """Return plan."""
        kind = resolve_disturbance_kind(cfg)
        handler = self.handlers.get(kind)
        if handler is None:
            raise ValueError(f"Unsupported disturbance kind: {kind}")
        return handler.plan(ss, cfg, rng)

    def apply(self, ss, spec: DisturbanceSpec, cfg: Dict, rng: np.random.Generator) -> None:
        """Return apply."""
        handler = self.handlers.get(spec.kind)
        if handler is None:
            raise ValueError(f"No handler registered for disturbance kind: {spec.kind}")
        handler.apply(ss, spec, cfg, rng)
