from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research_dataset import load_canonical_research_dataset
from universe import ALL_INSTRUMENTS


def build_tensor_pack(dataset) -> dict[str, np.ndarray]:
    pack: dict[str, np.ndarray] = {}
    for tf_name in dataset.timeframes:
        symbols = [symbol for symbol in ALL_INSTRUMENTS if symbol in dataset.tf_data[tf_name]]
        if not symbols:
            continue
        common_len = min(len(dataset.tf_data[tf_name][symbol]["c"]) for symbol in symbols)
        tensor = np.zeros((common_len, len(symbols), 6), dtype=np.float32)
        for sym_idx, symbol in enumerate(symbols):
            frame = dataset.tf_data[tf_name][symbol]
            tensor[:, sym_idx, 0] = frame["o"][-common_len:]
            tensor[:, sym_idx, 1] = frame["h"][-common_len:]
            tensor[:, sym_idx, 2] = frame["l"][-common_len:]
            tensor[:, sym_idx, 3] = frame["c"][-common_len:]
            tensor[:, sym_idx, 4] = frame["sp"][-common_len:]
            tensor[:, sym_idx, 5] = frame["tk"][-common_len:]
        pack[tf_name] = tensor
    return pack


def main():
    parser = argparse.ArgumentParser(description="Export canonical STGNN tensor pack")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data/research_stgnn_pack.npz")
    parser.add_argument("--meta-output", default="data/research_stgnn_pack_meta.json")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    dataset = load_canonical_research_dataset(args.data_dir, symbols=ALL_INSTRUMENTS, start=args.start, end=args.end)
    pack = build_tensor_pack(dataset)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)

    meta = {
        "timeframes": list(pack.keys()),
        "symbols": list(ALL_INSTRUMENTS),
        "outer_holdout_quarters": list(dataset.outer_holdout_quarters),
        "n_bars_per_tf": {tf: int(arr.shape[0]) for tf, arr in pack.items()},
        "feature_order": ["o", "h", "l", "c", "sp", "tk"],
    }
    with open(args.meta_output, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved STGNN tensor pack to {args.output}")
    print(f"Timeframes={list(pack.keys())}")


if __name__ == "__main__":
    main()
