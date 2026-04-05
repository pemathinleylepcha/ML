from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import shutil

import numpy as np

from staged_v4.config import FX_TRADABLE_NAMES
from staged_v4.contracts import BTCFeatureBatch, FXFeatureBatch, TimeframeFeatureBatch
from staged_v4.data.bridge_features import build_bridge_batches
from staged_v4.data.btc_features import build_btc_feature_batch, build_btc_timeframe_batch
from staged_v4.data.dataset import StagedPanels, generate_synthetic_panels, load_staged_panels
from staged_v4.data.fx_features import build_fx_feature_batch, build_fx_timeframe_batch
from staged_v4.utils.runtime_logging import stage_context, write_status


def _serialize_timeframe_batch(tf_batch: TimeframeFeatureBatch, out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        timestamps=tf_batch.timestamps,
        tradable_mask=tf_batch.tradable_mask,
        node_features=tf_batch.node_features,
        tpo_features=tf_batch.tpo_features,
        volatility=tf_batch.volatility,
        valid_mask=tf_batch.valid_mask,
        market_open_mask=tf_batch.market_open_mask,
        overlap_mask=tf_batch.overlap_mask,
        session_codes=tf_batch.session_codes,
        direction_labels=tf_batch.direction_labels,
        entry_labels=tf_batch.entry_labels,
        label_valid_mask=tf_batch.label_valid_mask,
    )


def _deserialize_timeframe_batch(timeframe: str, node_names: tuple[str, ...], payload_path: Path) -> TimeframeFeatureBatch:
    payload = np.load(payload_path, allow_pickle=False)
    return TimeframeFeatureBatch(
        timeframe=timeframe,
        timestamps=payload["timestamps"],
        node_names=node_names,
        tradable_mask=payload["tradable_mask"].astype(np.bool_),
        node_features=payload["node_features"].astype(np.float32),
        tpo_features=payload["tpo_features"].astype(np.float32),
        volatility=payload["volatility"].astype(np.float32),
        valid_mask=payload["valid_mask"].astype(np.bool_),
        market_open_mask=payload["market_open_mask"].astype(np.bool_),
        overlap_mask=payload["overlap_mask"].astype(np.bool_),
        session_codes=payload["session_codes"].astype(np.int8),
        direction_labels=payload["direction_labels"] if "direction_labels" in payload.files else None,
        entry_labels=payload["entry_labels"] if "entry_labels" in payload.files else None,
        label_valid_mask=payload["label_valid_mask"].astype(np.bool_) if "label_valid_mask" in payload.files else None,
    )


def _write_batch_metadata(
    batch,
    batch_root: Path,
    walkforward_splits: list[dict[str, object]],
    extra: dict[str, object] | None = None,
) -> None:
    metadata = {
        "anchor_timeframe": batch.anchor_timeframe,
        "anchor_timestamps": batch.anchor_timestamps.astype("datetime64[ns]").astype(str).tolist(),
        "node_names": list(batch.node_names),
        "timeframes": list(batch.timeframe_batches.keys()),
        "anchor_lookup": {k: v.tolist() for k, v in batch.anchor_lookup.items()},
        "walkforward_splits": [
            {
                **{k: v for k, v in split.items() if k not in {"train_idx", "val_idx"}},
                "train_idx": np.asarray(split["train_idx"], dtype=np.int32).tolist(),
                "val_idx": np.asarray(split["val_idx"], dtype=np.int32).tolist(),
            }
            for split in walkforward_splits
        ],
    }
    if hasattr(batch, "tradable_node_names"):
        metadata["tradable_node_names"] = list(batch.tradable_node_names)
    if extra:
        metadata.update(extra)
    (batch_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_batch_metadata(batch_root: Path) -> dict:
    return json.loads((batch_root / "metadata.json").read_text(encoding="utf-8"))


def _write_batch_metadata_from_parts(
    batch_root: Path,
    anchor_timeframe: str,
    anchor_timestamps: np.ndarray,
    node_names: tuple[str, ...],
    timeframes: tuple[str, ...],
    anchor_lookup: dict[str, np.ndarray],
    walkforward_splits: list[dict[str, object]],
    split_frequency: str,
    tradable_node_names: tuple[str, ...] | None = None,
) -> None:
    metadata = {
        "anchor_timeframe": anchor_timeframe,
        "anchor_timestamps": anchor_timestamps.astype("datetime64[ns]").astype(str).tolist(),
        "node_names": list(node_names),
        "timeframes": list(timeframes),
        "anchor_lookup": {k: v.tolist() for k, v in anchor_lookup.items() if k in timeframes},
        "walkforward_splits": [
            {
                **{k: v for k, v in split.items() if k not in {"train_idx", "val_idx"}},
                "train_idx": np.asarray(split["train_idx"], dtype=np.int32).tolist(),
                "val_idx": np.asarray(split["val_idx"], dtype=np.int32).tolist(),
            }
            for split in walkforward_splits
        ],
        "split_frequency": split_frequency,
    }
    if tradable_node_names is not None:
        metadata["tradable_node_names"] = list(tradable_node_names)
    (batch_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_feature_batches(
    output_root: str | Path,
    btc_batch: BTCFeatureBatch,
    fx_batch: FXFeatureBatch,
    walkforward_splits: list[dict[str, object]],
    split_frequency: str,
) -> dict:
    root = Path(output_root)
    btc_root = root / "btc"
    fx_root = root / "fx"
    bridge_root = root / "bridge"
    btc_root.mkdir(parents=True, exist_ok=True)
    fx_root.mkdir(parents=True, exist_ok=True)
    bridge_root.mkdir(parents=True, exist_ok=True)

    for timeframe, tf_batch in btc_batch.timeframe_batches.items():
        _serialize_timeframe_batch(tf_batch, btc_root / f"{timeframe}.npz")
    for timeframe, tf_batch in fx_batch.timeframe_batches.items():
        _serialize_timeframe_batch(tf_batch, fx_root / f"{timeframe}.npz")

    bridges = build_bridge_batches(btc_batch, fx_batch)
    bridge_meta = {
        timeframe: {
            "btc_index_for_fx": batch.btc_index_for_fx.tolist(),
            "overlap_mask": batch.overlap_mask.astype(np.int8).tolist(),
            "fx_timestamps": batch.fx_timestamps.astype("datetime64[ns]").astype(str).tolist(),
        }
        for timeframe, batch in bridges.items()
    }
    (bridge_root / "metadata.json").write_text(json.dumps(bridge_meta, indent=2), encoding="utf-8")

    _write_batch_metadata(btc_batch, btc_root, walkforward_splits, extra={"split_frequency": split_frequency})
    _write_batch_metadata(fx_batch, fx_root, walkforward_splits, extra={"split_frequency": split_frequency})

    manifest = {
        "root": str(root),
        "btc_timeframes": list(btc_batch.timeframe_batches.keys()),
        "fx_timeframes": list(fx_batch.timeframe_batches.keys()),
        "split_frequency": split_frequency,
        "walkforward_folds": len(walkforward_splits),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_feature_batches(cache_root: str | Path) -> tuple[BTCFeatureBatch, FXFeatureBatch, list[dict[str, object]], dict[str, object]]:
    root = Path(cache_root)
    btc_root = root / "btc"
    fx_root = root / "fx"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {}
    btc_meta = _read_batch_metadata(btc_root)
    fx_meta = _read_batch_metadata(fx_root)

    def _load_batch(batch_root: Path, meta: dict, batch_cls):
        node_names = tuple(meta["node_names"])
        timeframe_batches = {
            timeframe: _deserialize_timeframe_batch(timeframe, node_names, batch_root / f"{timeframe}.npz")
            for timeframe in meta["timeframes"]
        }
        anchor_lookup = {k: np.asarray(v, dtype=np.int32) for k, v in meta["anchor_lookup"].items()}
        anchor_timestamps = np.asarray(meta["anchor_timestamps"], dtype="datetime64[ns]")
        kwargs = {
            "anchor_timeframe": meta["anchor_timeframe"],
            "anchor_timestamps": anchor_timestamps,
            "timeframe_batches": timeframe_batches,
            "node_names": node_names,
            "anchor_lookup": anchor_lookup,
        }
        if "tradable_node_names" in meta:
            kwargs["tradable_node_names"] = tuple(meta["tradable_node_names"])
        return batch_cls(**kwargs)

    btc_batch = _load_batch(btc_root, btc_meta, BTCFeatureBatch)
    fx_batch = _load_batch(fx_root, fx_meta, FXFeatureBatch)
    stored_splits = fx_meta.get("walkforward_splits", fx_meta.get("monthly_splits", []))
    walkforward_splits = [
        {
            **{k: v for k, v in split.items() if k not in {"train_idx", "val_idx"}},
            "train_idx": np.asarray(split["train_idx"], dtype=np.int32),
            "val_idx": np.asarray(split["val_idx"], dtype=np.int32),
        }
        for split in stored_splits
    ]
    split_meta = {
        "split_frequency": fx_meta.get("split_frequency", "month"),
        "walkforward_folds": len(walkforward_splits),
        "outer_holdout_blocks": manifest.get("outer_holdout_blocks"),
    }
    return btc_batch, fx_batch, walkforward_splits, split_meta


def prepare_staged_cache(
    output_root: str | Path,
    mode: str,
    candle_root: str | None = None,
    tick_root: str | None = None,
    start: str | None = None,
    end: str | None = None,
    anchor_timeframe: str = "M1",
    strict: bool = False,
    timeframes: tuple[str, ...] | None = None,
    synthetic_n_anchor: int = 96,
    include_signal_only_tpo: bool = True,
    split_frequency: str = "week",
    outer_holdout_blocks: int = 1,
    min_train_blocks: int = 2,
    purge_bars: int = 6,
    max_workers: int = 0,
    logger: logging.Logger | None = None,
    status_file: str | None = None,
) -> dict:
    logger = logger or logging.getLogger("prepare_staged_cache")
    if mode == "synthetic":
        with stage_context(logger, status_file, "generate_synthetic_panels", n_anchor=synthetic_n_anchor, anchor_timeframe=anchor_timeframe):
            btc_panels, fx_panels = generate_synthetic_panels(n_anchor=synthetic_n_anchor, anchor_timeframe=anchor_timeframe)
    else:
        if candle_root is None:
            raise ValueError("candle_root is required in real mode")
        with stage_context(
            logger,
            status_file,
            "load_staged_panels",
            candle_root=candle_root,
            tick_root=tick_root,
            start=start,
            end=end,
            timeframes=timeframes,
            max_workers=max_workers,
        ):
            btc_panels, fx_panels = load_staged_panels(
                candle_root=candle_root,
                tick_root=tick_root,
                start=start,
                end=end,
                anchor_timeframe=anchor_timeframe,
                strict=strict,
                timeframes=timeframes,
                split_frequency=split_frequency,
                outer_holdout_blocks=outer_holdout_blocks,
                min_train_blocks=min_train_blocks,
                purge_bars=purge_bars,
                logger=logger,
                status_file=status_file,
                max_workers=max_workers,
            )
    selected_timeframes = tuple(timeframes or fx_panels.panels.keys())
    root = Path(output_root)
    btc_root = root / "btc"
    fx_root = root / "fx"
    bridge_root = root / "bridge"
    btc_root.mkdir(parents=True, exist_ok=True)
    fx_root.mkdir(parents=True, exist_ok=True)
    bridge_root.mkdir(parents=True, exist_ok=True)

    fx_bridge_meta: dict[str, dict[str, object]] = {}
    built_btc_timeframes: list[str] = []
    built_fx_timeframes: list[str] = []

    for timeframe in selected_timeframes:
        btc_tf_path = btc_root / f"{timeframe}.npz"
        if btc_tf_path.exists():
            logger.info("stage=build_btc_feature_timeframe state=skip timeframe=%s path=%s", timeframe, btc_tf_path.name)
        else:
            with stage_context(logger, status_file, "build_btc_feature_timeframe", timeframe=timeframe):
                btc_tf_batch = build_btc_timeframe_batch(btc_panels, timeframe, logger=logger, status_file=status_file)
            _serialize_timeframe_batch(btc_tf_batch, btc_tf_path)
            del btc_tf_batch
            gc.collect()
        built_btc_timeframes.append(timeframe)

        fx_tf_path = fx_root / f"{timeframe}.npz"
        shard_root = fx_root / "_shards" / timeframe
        if fx_tf_path.exists():
            logger.info("stage=build_fx_feature_timeframe state=skip timeframe=%s path=%s", timeframe, fx_tf_path.name)
            fx_tf_batch = _deserialize_timeframe_batch(timeframe, fx_panels.symbols, fx_tf_path)
        else:
            with stage_context(
                logger,
                status_file,
                "build_fx_feature_timeframe",
                timeframe=timeframe,
                include_signal_only_tpo=include_signal_only_tpo,
            ):
                fx_tf_batch = build_fx_timeframe_batch(
                    fx_panels,
                    timeframe,
                    include_signal_only_tpo=include_signal_only_tpo,
                    max_workers=max_workers,
                    logger=logger,
                    status_file=status_file,
                    shard_root=shard_root if timeframe == "tick" else None,
                )
            _serialize_timeframe_batch(fx_tf_batch, fx_tf_path)
            if shard_root.exists():
                shutil.rmtree(shard_root, ignore_errors=True)
        built_fx_timeframes.append(timeframe)
        btc_anchor = np.asarray(btc_panels.anchor_timestamps, dtype="datetime64[ns]")
        fx_ts = np.asarray(fx_tf_batch.timestamps, dtype="datetime64[ns]")
        btc_idx = btc_anchor.searchsorted(fx_ts, side="right") - 1
        btc_idx = np.maximum(btc_idx, 0).astype(np.int32)
        fx_bridge_meta[timeframe] = {
            "btc_index_for_fx": btc_idx.tolist(),
            "overlap_mask": np.asarray(fx_tf_batch.overlap_mask, dtype=np.bool_).astype(np.int8).tolist(),
            "fx_timestamps": fx_ts.astype("datetime64[ns]").astype(str).tolist(),
        }
        del fx_tf_batch, btc_idx, fx_ts
        gc.collect()

    with stage_context(logger, status_file, "write_streaming_cache_metadata", output_root=str(output_root)):
        (bridge_root / "metadata.json").write_text(json.dumps(fx_bridge_meta, indent=2), encoding="utf-8")
        _write_batch_metadata_from_parts(
            btc_root,
            btc_panels.anchor_timeframe,
            btc_panels.anchor_timestamps,
            btc_panels.symbols,
            tuple(built_btc_timeframes),
            btc_panels.anchor_lookup,
            fx_panels.walkforward_splits,
            fx_panels.split_frequency,
        )
        _write_batch_metadata_from_parts(
            fx_root,
            fx_panels.anchor_timeframe,
            fx_panels.anchor_timestamps,
            fx_panels.symbols,
            tuple(built_fx_timeframes),
            fx_panels.anchor_lookup,
            fx_panels.walkforward_splits,
            fx_panels.split_frequency,
            tradable_node_names=FX_TRADABLE_NAMES,
        )
        manifest = {
            "root": str(root),
            "btc_timeframes": list(built_btc_timeframes),
            "fx_timeframes": list(built_fx_timeframes),
            "split_frequency": fx_panels.split_frequency,
            "walkforward_folds": len(fx_panels.walkforward_splits),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest.update(
        {
            "mode": mode,
            "anchor_timeframe": anchor_timeframe,
            "start": start,
            "end": end,
            "strict": strict,
            "include_signal_only_tpo": include_signal_only_tpo,
            "outer_holdout_blocks": outer_holdout_blocks,
            "min_train_blocks": min_train_blocks,
            "purge_bars": purge_bars,
            "max_workers": max_workers,
        }
    )
    (Path(output_root) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status(status_file, {"state": "completed", "stage": "prepare_staged_cache", "manifest": manifest})
    return manifest
