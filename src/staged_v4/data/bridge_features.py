from __future__ import annotations

import numpy as np

from staged_v4.contracts import BTCFeatureBatch, BridgeBatch, FXFeatureBatch


def build_bridge_batches(btc_batch: BTCFeatureBatch, fx_batch: FXFeatureBatch) -> dict[str, BridgeBatch]:
    batches: dict[str, BridgeBatch] = {}
    btc_anchor = np.asarray(btc_batch.anchor_timestamps, dtype="datetime64[ns]")
    for timeframe, fx_tf in fx_batch.timeframe_batches.items():
        fx_ts = np.asarray(fx_tf.timestamps, dtype="datetime64[ns]")
        btc_idx = btc_anchor.searchsorted(fx_ts, side="right") - 1
        btc_idx = np.maximum(btc_idx, 0).astype(np.int32)
        batches[timeframe] = BridgeBatch(
            timeframe=timeframe,
            fx_timestamps=fx_ts,
            btc_index_for_fx=btc_idx,
            overlap_mask=np.asarray(fx_tf.overlap_mask, dtype=np.bool_),
        )
    return batches
