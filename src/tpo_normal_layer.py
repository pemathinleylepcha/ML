from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from staged_v4.config import ATR_MIN_THRESHOLD, ATR_NORM_CLIP


EPS = 1e-8


@dataclass(slots=True)
class TPOProfile:
    poc: float
    value_area_low: float
    value_area_high: float
    profile_low: float
    profile_high: float
    balance_score: float
    distance_to_poc_atr: float
    value_area_width_atr: float
    lookback: int = 0
    weight: float = 1.0
    degenerate: bool = False


@dataclass(slots=True)
class TPOMemoryState:
    composite_profile: TPOProfile
    profiles: tuple[TPOProfile, ...]
    support_score: float
    resistance_score: float
    rejection_score: float
    poc_drift_atr: float
    value_area_overlap: float
    degenerate: bool = False


@dataclass(slots=True)
class TPONormalDecision:
    direction: int
    confidence: float
    protector_blocked: bool
    protector_reason: str
    tp_distance: float
    sl_distance: float
    lot_scale: float
    profile: TPOProfile
    memory: TPOMemoryState
    legacy_direction: int
    legacy_confidence: float


def is_degenerate_atr(atr_price: float) -> bool:
    return not np.isfinite(float(atr_price)) or float(atr_price) <= ATR_MIN_THRESHOLD


def _zero_tpo_profile(last_price: float, *, lookback: int = 0, weight: float = 1.0, degenerate: bool = False) -> TPOProfile:
    return TPOProfile(
        poc=last_price,
        value_area_low=last_price,
        value_area_high=last_price,
        profile_low=last_price,
        profile_high=last_price,
        balance_score=0.0,
        distance_to_poc_atr=0.0,
        value_area_width_atr=0.0,
        lookback=lookback,
        weight=weight,
        degenerate=degenerate,
    )


def _zero_tpo_memory_state(last_price: float, *, lookback: int = 0, degenerate: bool = False) -> TPOMemoryState:
    profile = _zero_tpo_profile(last_price, lookback=lookback, degenerate=degenerate)
    return TPOMemoryState(
        composite_profile=profile,
        profiles=(profile,),
        support_score=0.0,
        resistance_score=0.0,
        rejection_score=0.0,
        poc_drift_atr=0.0,
        value_area_overlap=0.0,
        degenerate=degenerate,
    )


def _build_tpo_counts(
    high: np.ndarray,
    low: np.ndarray,
    profile_low: float,
    profile_high: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(profile_low, profile_high, n_bins + 1, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    for lo, hi in zip(low.astype(np.float64, copy=False), high.astype(np.float64, copy=False)):
        lo_idx = int(np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, n_bins - 1))
        hi_idx = int(np.clip(np.searchsorted(edges, hi, side="left"), 1, n_bins))
        counts[lo_idx:hi_idx] += 1.0

    return counts, edges


def _value_area_bounds(counts: np.ndarray, edges: np.ndarray, value_area_fraction: float) -> tuple[float, float, float]:
    total = float(np.sum(counts))
    if total <= 0.0:
        mid = 0.5 * (edges[0] + edges[-1])
        return mid, edges[0], edges[-1]

    poc_idx = int(np.argmax(counts))
    included = {poc_idx}
    cumulative = float(counts[poc_idx])
    left = poc_idx - 1
    right = poc_idx + 1
    target = total * float(np.clip(value_area_fraction, 0.1, 1.0))

    while cumulative < target and (left >= 0 or right < len(counts)):
        left_value = counts[left] if left >= 0 else -1.0
        right_value = counts[right] if right < len(counts) else -1.0
        if right_value >= left_value:
            if right < len(counts):
                included.add(right)
                cumulative += float(counts[right])
                right += 1
            elif left >= 0:
                included.add(left)
                cumulative += float(counts[left])
                left -= 1
        else:
            if left >= 0:
                included.add(left)
                cumulative += float(counts[left])
                left -= 1
            elif right < len(counts):
                included.add(right)
                cumulative += float(counts[right])
                right += 1

    min_idx = min(included)
    max_idx = max(included)
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])
    value_area_low = float(edges[min_idx])
    value_area_high = float(edges[max_idx + 1])
    return float(poc), value_area_low, value_area_high


def compute_tpo_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_price: float,
    lookback: int = 48,
    n_bins: int = 24,
    value_area_fraction: float = 0.70,
) -> TPOProfile:
    if len(close) == 0:
        raise ValueError("TPO profile requires at least one close value")

    tail_high = high[-lookback:].astype(np.float64, copy=False)
    tail_low = low[-lookback:].astype(np.float64, copy=False)
    tail_close = close[-lookback:].astype(np.float64, copy=False)
    effective_lookback = int(min(lookback, len(close)))
    last_price = float(tail_close[-1])

    if is_degenerate_atr(atr_price):
        return _zero_tpo_profile(last_price, lookback=effective_lookback, degenerate=True)

    profile_low = float(np.min(tail_low))
    profile_high = float(np.max(tail_high))
    if profile_high - profile_low < EPS:
        return TPOProfile(
            poc=last_price,
            value_area_low=last_price,
            value_area_high=last_price,
            profile_low=last_price,
            profile_high=last_price,
            balance_score=1.0,
            distance_to_poc_atr=0.0,
            value_area_width_atr=0.0,
            lookback=effective_lookback,
        )

    counts, edges = _build_tpo_counts(tail_high, tail_low, profile_low, profile_high, n_bins=n_bins)
    poc, value_area_low, value_area_high = _value_area_bounds(counts, edges, value_area_fraction=value_area_fraction)
    balance_score = float(np.max(counts) / max(np.sum(counts), 1.0))
    atr_ref = float(atr_price)

    return TPOProfile(
        poc=float(poc),
        value_area_low=float(value_area_low),
        value_area_high=float(value_area_high),
        profile_low=float(profile_low),
        profile_high=float(profile_high),
        balance_score=balance_score,
        distance_to_poc_atr=float(np.clip((last_price - poc) / atr_ref, -ATR_NORM_CLIP, ATR_NORM_CLIP)),
        value_area_width_atr=float(np.clip((value_area_high - value_area_low) / atr_ref, 0.0, ATR_NORM_CLIP)),
        lookback=effective_lookback,
        weight=1.0,
    )


def _value_area_overlap(profile_a: TPOProfile, profile_b: TPOProfile) -> float:
    left = max(profile_a.value_area_low, profile_b.value_area_low)
    right = min(profile_a.value_area_high, profile_b.value_area_high)
    overlap = max(0.0, right - left)
    width = max(
        profile_a.value_area_high - profile_a.value_area_low,
        profile_b.value_area_high - profile_b.value_area_low,
        EPS,
    )
    return float(overlap / width)


def compute_tpo_memory_state(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_price: float,
    lookbacks: tuple[int, ...] = (24, 48, 96, 192),
    n_bins: int = 32,
    value_area_fraction: float = 0.70,
    rejection_lookback: int = 12,
) -> TPOMemoryState:
    if len(close) == 0:
        raise ValueError("TPO memory state requires at least one close value")
    if is_degenerate_atr(atr_price):
        return _zero_tpo_memory_state(float(close[-1]), lookback=min(max(lookbacks, default=0), len(close)), degenerate=True)

    profiles: list[TPOProfile] = []
    raw_weights: list[float] = []
    for lookback in lookbacks:
        if len(close) < min(lookback, 12):
            continue
        effective = min(int(lookback), len(close))
        profile = compute_tpo_profile(
            high=high,
            low=low,
            close=close,
            atr_price=atr_price,
            lookback=effective,
            n_bins=n_bins,
            value_area_fraction=value_area_fraction,
        )
        profiles.append(profile)
        raw_weights.append(1.0 / np.sqrt(float(effective)))

    if not profiles:
        profile = compute_tpo_profile(
            high=high,
            low=low,
            close=close,
            atr_price=atr_price,
            lookback=min(len(close), 12),
            n_bins=n_bins,
            value_area_fraction=value_area_fraction,
        )
        return TPOMemoryState(
            composite_profile=profile,
            profiles=(profile,),
            support_score=0.0,
            resistance_score=0.0,
            rejection_score=0.0,
            poc_drift_atr=0.0,
            value_area_overlap=1.0,
        )

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights /= max(float(weights.sum()), EPS)
    weighted_profiles = tuple(
        TPOProfile(
            poc=profile.poc,
            value_area_low=profile.value_area_low,
            value_area_high=profile.value_area_high,
            profile_low=profile.profile_low,
            profile_high=profile.profile_high,
            balance_score=profile.balance_score,
            distance_to_poc_atr=profile.distance_to_poc_atr,
            value_area_width_atr=profile.value_area_width_atr,
            lookback=profile.lookback,
            weight=float(weight),
            degenerate=profile.degenerate,
        )
        for profile, weight in zip(profiles, weights)
    )

    def _weighted_avg(attr: str) -> float:
        return float(sum(getattr(profile, attr) * profile.weight for profile in weighted_profiles))

    composite = TPOProfile(
        poc=_weighted_avg("poc"),
        value_area_low=_weighted_avg("value_area_low"),
        value_area_high=_weighted_avg("value_area_high"),
        profile_low=min(profile.profile_low for profile in weighted_profiles),
        profile_high=max(profile.profile_high for profile in weighted_profiles),
        balance_score=_weighted_avg("balance_score"),
        distance_to_poc_atr=_weighted_avg("distance_to_poc_atr"),
        value_area_width_atr=_weighted_avg("value_area_width_atr"),
        lookback=max(profile.lookback for profile in weighted_profiles),
        weight=1.0,
    )

    last_price = float(close[-1])
    support_score = float(
        sum(profile.weight for profile in weighted_profiles if last_price < profile.value_area_low)
    )
    resistance_score = float(
        sum(profile.weight for profile in weighted_profiles if last_price > profile.value_area_high)
    )
    recent_close = close[-min(int(rejection_lookback), len(close)) :].astype(np.float64, copy=False)
    in_value = (recent_close >= composite.value_area_low) & (recent_close <= composite.value_area_high)
    rejection_score = float(1.0 - np.mean(in_value)) if len(recent_close) > 0 else 0.0
    atr_ref = float(atr_price)
    poc_drift_atr = 0.0
    if len(weighted_profiles) >= 2:
        poc_drift_atr = float(np.clip((weighted_profiles[0].poc - weighted_profiles[-1].poc) / atr_ref, -ATR_NORM_CLIP, ATR_NORM_CLIP))
    overlap_scores = []
    for left, right in zip(weighted_profiles[:-1], weighted_profiles[1:]):
        overlap_scores.append(_value_area_overlap(left, right))
    value_area_overlap = float(np.mean(overlap_scores)) if overlap_scores else 1.0

    return TPOMemoryState(
        composite_profile=composite,
        profiles=weighted_profiles,
        support_score=support_score,
        resistance_score=resistance_score,
        rejection_score=rejection_score,
        poc_drift_atr=poc_drift_atr,
        value_area_overlap=value_area_overlap,
    )


def build_tpo_normal_decision(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_price: float,
    spread_price: float,
    legacy_direction: int,
    legacy_confidence: float,
    legacy_p_buy: float,
    legacy_p_sell: float,
    reversal_lookback: int = 48,
    profile_bins: int = 32,
    memory_lookbacks: tuple[int, ...] = (24, 48, 96, 192),
    entry_outside_value_area_atr: float = 0.08,
    protector_confidence: float = 0.14,
    protector_atr_norm: float = 0.0017,
    protector_spread_to_atr: float = 0.22,
) -> TPONormalDecision:
    if is_degenerate_atr(atr_price):
        memory = _zero_tpo_memory_state(float(close[-1]), lookback=min(reversal_lookback, len(close)), degenerate=True)
        profile = memory.composite_profile
        fallback_distance = max(spread_price, ATR_MIN_THRESHOLD)
        return TPONormalDecision(
            direction=0,
            confidence=0.0,
            protector_blocked=False,
            protector_reason="degenerate_atr",
            tp_distance=fallback_distance,
            sl_distance=fallback_distance,
            lot_scale=0.0,
            profile=profile,
            memory=memory,
            legacy_direction=int(legacy_direction),
            legacy_confidence=float(legacy_confidence),
        )
    if len(close) < max(reversal_lookback, 8):
        profile = compute_tpo_profile(high, low, close, atr_price=atr_price, lookback=len(close))
        memory = TPOMemoryState(
            composite_profile=profile,
            profiles=(profile,),
            support_score=0.0,
            resistance_score=0.0,
            rejection_score=0.0,
            poc_drift_atr=0.0,
            value_area_overlap=1.0,
        )
        return TPONormalDecision(
            direction=0,
            confidence=0.0,
            protector_blocked=False,
            protector_reason="insufficient_history",
            tp_distance=max(atr_price, spread_price),
            sl_distance=max(0.75 * atr_price, spread_price),
            lot_scale=0.0,
            profile=profile,
            memory=memory,
            legacy_direction=int(legacy_direction),
            legacy_confidence=float(legacy_confidence),
        )

    atr_ref = float(atr_price)
    last_price = float(close[-1])
    last_ret = float(np.log(max(close[-1], EPS) / max(close[-2], EPS)))
    medium_ret = float(np.log(max(close[-1], EPS) / max(close[-4], EPS)))
    atr_norm = atr_ref / max(abs(last_price), EPS)
    spread_to_atr = float(spread_price / atr_ref) if atr_ref > 0.0 else 1.0

    memory = compute_tpo_memory_state(
        high=high,
        low=low,
        close=close,
        atr_price=atr_ref,
        n_bins=profile_bins,
        lookbacks=memory_lookbacks,
    )
    profile = memory.composite_profile

    below_value_area = last_price < (profile.value_area_low - entry_outside_value_area_atr * atr_ref)
    above_value_area = last_price > (profile.value_area_high + entry_outside_value_area_atr * atr_ref)
    bullish_reversal = last_ret > 0.0 and medium_ret > last_ret * 0.25 and memory.support_score > 0.10
    bearish_reversal = last_ret < 0.0 and medium_ret < last_ret * 0.25 and memory.resistance_score > 0.10
    protector_reason = ""
    protector_blocked = False

    # Legacy model acts as a volatility/trend guard for the new normal-trading layer.
    if legacy_confidence >= protector_confidence:
        protector_blocked = True
        protector_reason = "legacy_high_confidence"
    if atr_norm >= protector_atr_norm:
        protector_blocked = True
        protector_reason = "atr_volatility"
    if spread_to_atr >= protector_spread_to_atr:
        protector_blocked = True
        protector_reason = "spread_stress"

    direction = 0
    confidence = 0.0
    lot_scale = 0.0
    tp_distance = max(0.90 * atr_ref, spread_price * 4.0)
    sl_distance = max(0.65 * atr_ref, spread_price * 3.0)
    deep_support = min(memory.composite_profile.value_area_low, *(profile_.value_area_low for profile_ in memory.profiles))
    deep_resistance = max(memory.composite_profile.value_area_high, *(profile_.value_area_high for profile_ in memory.profiles))
    long_horizon_poc = max(memory.profiles, key=lambda profile_: profile_.lookback).poc
    memory_anchor_gap = abs(long_horizon_poc - profile.poc)

    if below_value_area and bullish_reversal:
        direction = 1
        distance_score = max((profile.value_area_low - last_price) / atr_ref, 0.0)
        memory_score = min(1.0, memory.support_score + 0.35 * memory.rejection_score)
        trend_penalty = max(-memory.poc_drift_atr - 0.85, 0.0)
        confidence = min(
            0.96,
            0.16
            + 0.24 * distance_score
            + 0.24 * abs(last_ret) / atr_ref
            + 0.16 * profile.balance_score
            + 0.16 * memory_score
            + 0.08 * max(memory.value_area_overlap, 0.0)
            - 0.10 * trend_penalty,
        )
        tp_distance = max(abs(profile.poc - last_price), abs(long_horizon_poc - last_price), 0.90 * atr_ref, spread_price * 4.0)
        sl_distance = max(abs(deep_support - last_price) * 0.60, 0.65 * atr_ref, spread_price * 3.0)
        lot_scale = float(np.clip(0.50 + 0.25 * distance_score + 0.20 * memory_score + 0.10 * memory_anchor_gap / atr_ref, 0.45, 1.35))
    elif above_value_area and bearish_reversal:
        direction = -1
        distance_score = max((last_price - profile.value_area_high) / atr_ref, 0.0)
        memory_score = min(1.0, memory.resistance_score + 0.35 * memory.rejection_score)
        trend_penalty = max(memory.poc_drift_atr - 0.85, 0.0)
        confidence = min(
            0.96,
            0.16
            + 0.24 * distance_score
            + 0.24 * abs(last_ret) / atr_ref
            + 0.16 * profile.balance_score
            + 0.16 * memory_score
            + 0.08 * max(memory.value_area_overlap, 0.0)
            - 0.10 * trend_penalty,
        )
        tp_distance = max(abs(last_price - profile.poc), abs(last_price - long_horizon_poc), 0.90 * atr_ref, spread_price * 4.0)
        sl_distance = max(abs(last_price - deep_resistance) * 0.60, 0.65 * atr_ref, spread_price * 3.0)
        lot_scale = float(np.clip(0.50 + 0.25 * distance_score + 0.20 * memory_score + 0.10 * memory_anchor_gap / atr_ref, 0.45, 1.35))

    if direction != 0 and legacy_direction == -direction and legacy_confidence >= protector_confidence * 0.75:
        protector_blocked = True
        protector_reason = "legacy_direction_conflict"

    if protector_blocked:
        direction = 0
        confidence = 0.0
        lot_scale = 0.0

    return TPONormalDecision(
        direction=int(direction),
        confidence=float(confidence),
        protector_blocked=bool(protector_blocked),
        protector_reason=protector_reason,
        tp_distance=float(tp_distance),
        sl_distance=float(sl_distance),
        lot_scale=float(lot_scale),
        profile=profile,
        memory=memory,
        legacy_direction=int(legacy_direction),
        legacy_confidence=float(legacy_confidence),
    )
