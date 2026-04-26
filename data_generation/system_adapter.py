# system_adapter.py
# Lightweight adapter over ANDES models for disturbance-generation workflows.

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class SystemAdapter:
    """Lightweight adapter for system-level ANDES interactions."""
    def __init__(self, ss) -> None:
        """Store the ANDES system handle."""
        self.ss = ss

    def pq_names(self) -> List[str]:
        """List PQ device names."""
        if not getattr(self.ss, "PQ", None) or not self.ss.PQ.n:
            return []
        return [str(name) for name in list(self.ss.PQ.name.v)]

    def pq_name_to_owner(self) -> Dict[str, str]:
        """Map PQ device names to owner labels."""
        if not getattr(self.ss, "PQ", None) or not self.ss.PQ.n:
            return {}
        return {str(name): str(owner) for name, owner in zip(self.ss.PQ.name.v, self.ss.PQ.owner.v)}

    def add_load_step(self, *, dev: str, time_s: float, scale: float) -> None:
        """Add active/reactive load-step events for one PQ device."""
        self.ss.add(
            model="Alter",
            param_dict=dict(
                t=float(time_s),
                model="PQ",
                dev=dev,
                src="Ppf",
                attr="v",
                method="*",
                amount=float(scale),
            ),
        )
        self.ss.add(
            model="Alter",
            param_dict=dict(
                t=float(time_s),
                model="PQ",
                dev=dev,
                src="Qpf",
                attr="v",
                method="*",
                amount=float(scale),
            ),
        )

    def add_line_toggle(self, *, dev, time_s: float) -> None:
        """Add a line toggle event."""
        self.ss.add(
            model="Toggle",
            param_dict={
                "t": float(time_s),
                "model": "Line",
                "dev": dev,
            },
        )

    def line_records(self) -> List[Dict[str, float | int | str]]:
        """Collect line records used by disturbance selection."""
        records: List[Dict[str, float | int | str]] = []
        n_line = int(getattr(self.ss.Line, "n", 0))
        idx_vals = list(getattr(getattr(self.ss.Line, "idx", None), "v", []))
        name_vals = list(getattr(getattr(self.ss.Line, "name", None), "v", []))
        bus1_vals = list(getattr(getattr(self.ss.Line, "bus1", None), "v", []))
        bus2_vals = list(getattr(getattr(self.ss.Line, "bus2", None), "v", []))
        u_vals = list(getattr(getattr(self.ss.Line, "u", None), "v", []))

        for uid in range(n_line):
            idx_val = idx_vals[uid] if uid < len(idx_vals) else uid + 1
            name_val = name_vals[uid] if uid < len(name_vals) else f"Line_{idx_val}"
            try:
                bus1 = float(bus1_vals[uid]) if uid < len(bus1_vals) else np.nan
            except Exception:
                bus1 = np.nan
            try:
                bus2 = float(bus2_vals[uid]) if uid < len(bus2_vals) else np.nan
            except Exception:
                bus2 = np.nan
            in_service = bool(u_vals[uid]) if uid < len(u_vals) else True
            records.append(
                {
                    "uid": uid,
                    "idx": idx_val,
                    "name": str(name_val),
                    "bus1": bus1,
                    "bus2": bus2,
                    "rating": self._line_rating(uid),
                    "in_service": in_service,
                }
            )
        return records

    def _line_rating(self, uid: int) -> float:
        """Read a line rating from available ANDES Line attributes."""
        for attr in ("rate_a", "rateA", "RATE_A"):
            obj = getattr(self.ss.Line, attr, None)
            if obj is None:
                continue
            vals = getattr(obj, "v", None)
            if vals is None or uid >= len(vals):
                continue
            try:
                return float(vals[uid])
            except Exception:
                return float("nan")
        return float("nan")
