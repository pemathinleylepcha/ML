from __future__ import annotations

import numpy as np

from staged_v5.config import RAW_FEATURE_NAMES, VP_FEATURE_NAMES, OF_FEATURE_NAMES
from staged_v5.execution_gate.contracts import GateEvent, direction_from_prob


_SP_INDEX = RAW_FEATURE_NAMES.index("sp")
_TK_INDEX = RAW_FEATURE_NAMES.index("tk")


def extract_tradable_anchor_views(
    fx_source,
    anchor_indices: np.ndarray,
    pair_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_tf = fx_source.anchor_timeframe
    tf_batch = fx_source.timeframe_batches[anchor_tf]
    lookup = {name: idx for idx, name in enumerate(tf_batch.node_names)}
    tradable_indices = np.asarray([lookup[name] for name in pair_names], dtype=np.int32)
    row_indices = np.asarray(anchor_indices, dtype=np.int32)
    timestamps = np.asarray(tf_batch.timestamps[row_indices])
    node_features = np.asarray(tf_batch.node_features[row_indices][:, tradable_indices], dtype=np.float32)
    tpo_features = np.asarray(tf_batch.tpo_features[row_indices][:, tradable_indices], dtype=np.float32)
    return timestamps, node_features, tpo_features


def build_gate_events(
    *,
    timestamps: np.ndarray,
    prob_buy: np.ndarray,
    prob_entry: np.ndarray | None,
    close: np.ndarray,
    atr: np.ndarray,
    volatility: np.ndarray,
    session_codes: np.ndarray,
    pair_names: tuple[str, ...],
    anchor_node_features: np.ndarray,
    anchor_tpo_features: np.ndarray,
    valid_mask: np.ndarray | None = None,
    vp_features_by_node: dict[int, np.ndarray] | None = None,
    of_features_by_node: dict[int, np.ndarray] | None = None,
) -> list[GateEvent]:
    _n_vp = len(VP_FEATURE_NAMES)
    _n_of = len(OF_FEATURE_NAMES)
    events: list[GateEvent] = []
    n_steps, n_nodes = prob_buy.shape
    for t in range(n_steps):
        if int(session_codes[t]) == 0:
            continue
        for node_idx in range(n_nodes):
            if valid_mask is not None and not bool(valid_mask[t, node_idx]):
                continue
            raw_prob_buy = float(prob_buy[t, node_idx])
            direction = direction_from_prob(raw_prob_buy)
            vp = None
            if vp_features_by_node is not None and node_idx in vp_features_by_node:
                vp_raw = vp_features_by_node[node_idx]
                if vp_raw.ndim == 2:
                    vp = np.asarray(vp_raw[t], dtype=np.float32)
                else:
                    vp = np.asarray(vp_raw, dtype=np.float32)
                if vp.shape != (_n_vp,):
                    vp = np.zeros(_n_vp, dtype=np.float32)
            of = None
            if of_features_by_node is not None and node_idx in of_features_by_node:
                of_raw = of_features_by_node[node_idx]
                if of_raw.ndim == 2:
                    of = np.asarray(of_raw[t], dtype=np.float32)
                else:
                    of = np.asarray(of_raw, dtype=np.float32)
                if of.shape != (_n_of,):
                    of = np.zeros(_n_of, dtype=np.float32)
            events.append(
                GateEvent(
                    timestamp=np.datetime64(timestamps[t]),
                    pair_name=pair_names[node_idx],
                    direction=direction,
                    prob_buy=raw_prob_buy,
                    prob_entry=float(prob_entry[t, node_idx]) if prob_entry is not None else None,
                    close=float(close[t, node_idx]),
                    atr=float(max(atr[t, node_idx], 1e-8)),
                    volatility=float(volatility[t, node_idx]),
                    session_code=int(session_codes[t]),
                    spread=float(anchor_node_features[t, node_idx, _SP_INDEX]),
                    tick_count=float(anchor_node_features[t, node_idx, _TK_INDEX]),
                    tpo_features=np.asarray(anchor_tpo_features[t, node_idx], dtype=np.float32),
                    vp_features=vp,
                    of_features=of,
                )
            )
    return events
