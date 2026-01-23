import pandas as pd
from typing import Optional, Sequence


def scale_pq_loads(
    ss,
    *,
    bus_ids: Optional[Sequence[int]] = None,
    pq_names: Optional[Sequence[str]] = None,
    p_scale: float = 1.0,
    q_scale: Optional[float] = None,
    t: Optional[float] = None,
):
    if ss.PQ.n == 0:
        return

    q_scale = p_scale if q_scale is None else q_scale
    alter_kwargs = {"t": float(t)} if t is not None else {}

    df = ss.PQ.as_df()
    mask = pd.Series(True, index=df.index)
    if bus_ids is not None:
        mask &= df["bus"].isin(bus_ids)
    if pq_names is not None:
        mask &= df["name"].isin(pq_names)

    if not mask.any():
        raise ValueError("No PQ loads matched the provided filters.")

    for uid, row in df.loc[mask].iterrows():
        idx = row["idx"]
        if t:
            ss.add(
                "Alter",
                dict(
                    model="PQ",
                    dev=idx,
                    src="Ppf",
                    attr="v",
                    method="*",
                    amount=p_scale,
                    **alter_kwargs,
                ),
            )
            ss.add(
                "Alter",
                dict(
                    model="PQ",
                    dev=idx,
                    src="Qpf",
                    attr="v",
                    method="*",
                    amount=q_scale,
                    **alter_kwargs,
                ),
            )
        else:
            ss.PQ.alter("Ppf", idx, float(ss.PQ.Ppf.v[uid]) * p_scale)
            ss.PQ.alter("Qpf", idx, float(ss.PQ.Qpf.v[uid]) * q_scale)


def scale_zip_loads(
    ss,
    *,
    bus_ids: Optional[Sequence[int]] = None,
    zip_names: Optional[Sequence[str]] = None,
    p_scale: float = 1.0,
    q_scale: Optional[float] = None,
    t: Optional[float] = None,
):
    if ss.ZIP.n == 0:
        return

    q_scale = p_scale if q_scale is None else q_scale
    alter_kwargs = {"t": float(t)} if t is not None else {}

    df = ss.ZIP.as_df()
    mask = pd.Series(True, index=df.index)
    if bus_ids is not None:
        mask &= df["bus"].isin(bus_ids)
    if zip_names is not None:
        mask &= df["name"].isin(zip_names)

    if not mask.any():
        raise ValueError("No ZIP loads matched the provided filters.")

    for _, row in df.loc[mask].iterrows():
        idx = row["idx"]
        ss.add(
            "Alter",
            dict(
                model="ZIP",
                dev=idx,
                src="pp0",
                attr="v",
                method="*",
                amount=p_scale,
                **alter_kwargs,
            ),
        )
        ss.add(
            "Alter",
            dict(
                model="ZIP",
                dev=idx,
                src="qp0",
                attr="v",
                method="*",
                amount=q_scale,
                **alter_kwargs,
            ),
        )
