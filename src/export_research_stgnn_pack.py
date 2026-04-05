from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_dataset import FILL_POLICY_MASK, SESSION_NAMES, encode_session_codes, load_canonical_research_dataset
from universe import ALL_INSTRUMENTS

FEATURE_ORDER = (
    "o",
    "h",
    "l",
    "c",
    "sp",
    "tk",
    "validity",
    "session_code",
    "session_transition",
    "regime_signal",
)


def _compute_session_transition(session_codes: np.ndarray) -> np.ndarray:
    transition = np.zeros(len(session_codes), dtype=np.float32)
    if len(session_codes) > 1:
        transition[1:] = (session_codes[1:] != session_codes[:-1]).astype(np.float32)
    return transition


def _compute_regime_signal(close: np.ndarray, valid: np.ndarray, lookback: int = 12) -> np.ndarray:
    close_series = pd.Series(close.astype(np.float64, copy=False))
    returns = close_series.pct_change(fill_method=None)
    trend = close_series.pct_change(lookback, fill_method=None)
    realized_vol = returns.abs().rolling(lookback, min_periods=3).mean()
    scaled = trend / (realized_vol * np.sqrt(float(lookback)) + 1e-8)
    regime = scaled.clip(-3.0, 3.0).fillna(0.0).to_numpy(dtype=np.float32) / 3.0
    regime[~valid] = 0.0
    return regime


def build_tensor_pack(dataset) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pack: dict[str, np.ndarray] = {}
    timestamps: dict[str, np.ndarray] = {}
    session_scale = float(max(1, len(SESSION_NAMES) - 1))
    for tf_name in dataset.timeframes:
        symbols = [symbol for symbol in ALL_INSTRUMENTS if symbol in dataset.tf_data[tf_name]]
        if not symbols:
            continue
        common_len = min(len(dataset.tf_data[tf_name][symbol]["c"]) for symbol in symbols)
        tensor = np.zeros((common_len, len(symbols), len(FEATURE_ORDER)), dtype=np.float32)
        tf_timestamps = dataset.tf_data[tf_name][symbols[0]]["dt"][-common_len:]
        tf_session_codes = encode_session_codes(tf_timestamps)
        tf_session_feature = tf_session_codes.astype(np.float32) / session_scale
        tf_session_transition = _compute_session_transition(tf_session_codes)
        for sym_idx, symbol in enumerate(symbols):
            frame = dataset.tf_data[tf_name][symbol]
            valid = frame["real"][-common_len:].astype(np.bool_, copy=False)
            tensor[:, sym_idx, 0] = frame["o"][-common_len:]
            tensor[:, sym_idx, 1] = frame["h"][-common_len:]
            tensor[:, sym_idx, 2] = frame["l"][-common_len:]
            tensor[:, sym_idx, 3] = frame["c"][-common_len:]
            tensor[:, sym_idx, 4] = frame["sp"][-common_len:]
            tensor[:, sym_idx, 5] = frame["tk"][-common_len:]
            tensor[:, sym_idx, 6] = valid.astype(np.float32)
            tensor[:, sym_idx, 7] = tf_session_feature
            tensor[:, sym_idx, 8] = tf_session_transition
            tensor[:, sym_idx, 9] = _compute_regime_signal(frame["c"][-common_len:], valid)
        pack[tf_name] = tensor
        timestamps[tf_name] = tf_timestamps.astype("datetime64[ns]").astype(np.int64)
    return pack, timestamps


def main():
    parser = argparse.ArgumentParser(description="Export canonical STGNN tensor pack")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data/research_stgnn_pack.npz")
    parser.add_argument("--meta-output", default="data/research_stgnn_pack_meta.json")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    dataset = load_canonical_research_dataset(
        args.data_dir,
        symbols=ALL_INSTRUMENTS,
        start=args.start,
        end=args.end,
        fill_policy=FILL_POLICY_MASK,
    )
    pack, timestamps = build_tensor_pack(dataset)
    session_scale = float(max(1, len(SESSION_NAMES) - 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_payload = {f"{tf}_ts": arr for tf, arr in timestamps.items()}
    np.savez_compressed(out_path, **pack, **timestamp_payload)

    meta = {
        "timeframes": list(pack.keys()),
        "symbols": list(ALL_INSTRUMENTS),
        "outer_holdout_quarters": list(dataset.outer_holdout_quarters),
        "n_bars_per_tf": {tf: int(arr.shape[0]) for tf, arr in pack.items()},
        "feature_order": list(FEATURE_ORDER),
        "fill_policy": dataset.fill_policy,
        "timestamp_keys": {tf: f"{tf}_ts" for tf in timestamps},
        "session_code_scale": session_scale,
    }
    with open(args.meta_output, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved STGNN tensor pack to {args.output}")
    print(f"Timeframes={list(pack.keys())}")


if __name__ == "__main__":
    main()
