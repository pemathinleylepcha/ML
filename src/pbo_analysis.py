"""
pbo_analysis.py -- Backtest Overfitting Diagnostics

Implements:
  - Walk-Forward Efficiency (WFE): OOS/IS Sharpe ratio
  - Probabilistic Sharpe Ratio (PSR): P(SR > SR*) per fold
  - Deflated Sharpe Ratio (DSR): PSR adjusted for multiple testing across folds
  - PBO via CSCV (Combinatorially Symmetric Cross-Validation):
      Each fold model is treated as a distinct "strategy variant".
      Generates C(n_folds, n_folds//2) IS/OOS splits; counts how often
      the best IS model underperforms OOS median (overfitting).

Reference: Bailey, Borwein, Lopez de Prado, Zhu (2014)
           "The Probability of Backtest Overfitting"
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List

import numpy as np

try:
    from scipy.stats import norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _phi(x: float) -> float:
    """Standard normal CDF — uses scipy if available, else Abramowitz & Stegun."""
    if _HAS_SCIPY:
        return float(_norm.cdf(x))
    # Rational approximation (max error 7.5e-8)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782
               + t * (1.781477937 + t * (-1.821255978
               + t * 1.330274429))))
    p = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return p if x >= 0 else 1.0 - p


def _sharpe(returns: np.ndarray) -> float:
    """Raw (non-annualized) Sharpe ratio."""
    std = float(returns.std())
    if std < 1e-12:
        return 0.0
    return float(returns.mean()) / std


def _moments(returns: np.ndarray):
    """Returns (mean, std, skew, excess_kurt) of a return series."""
    n = len(returns)
    if n < 4:
        return returns.mean(), returns.std(), 0.0, 0.0
    m = returns.mean()
    s = returns.std(ddof=1) + 1e-12
    skew = float(((returns - m) ** 3).mean() / s ** 3)
    kurt = float(((returns - m) ** 4).mean() / s ** 4) - 3.0
    return float(m), float(s), skew, kurt


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def psr(sr_hat: float, T: int, skew: float, excess_kurt: float,
        sr_star: float = 0.0) -> float:
    """
    PSR(SR*) = P(SR > SR* | sr_hat, T, skew, kurt)

    Bailey & Lopez de Prado (2012).  Returns value in [0, 1].
    """
    if T <= 1:
        return 0.5
    denom = math.sqrt(
        max(1e-12, 1.0 - skew * sr_hat + (excess_kurt - 1) / 4.0 * sr_hat ** 2)
    )
    z = (sr_hat - sr_star) * math.sqrt(T - 1) / denom
    return _phi(z)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def dsr(sharpe_values: List[float], n_bars_values: List[int],
        sr_star: float = 0.0) -> float:
    """
    DSR = PSR adjusted for multiple testing.

    Estimates the expected maximum SR under the null and uses it as SR*.
    Formula: E[max SR] ≈ median-bias-corrected expected maximum of N i.i.d. N(0,1) /sqrt(T).
    """
    N = len(sharpe_values)
    if N == 0:
        return 0.5

    T_bar = float(np.mean(n_bars_values)) if n_bars_values else 252
    T_bar = max(T_bar, 2)

    # Expected maximum of N i.i.d. standard normal draws (Bloch-Watson approx)
    if N == 1:
        e_max = 0.0
    else:
        euler_gamma = 0.5772156649
        e_max = (
            (1.0 - euler_gamma) * _phi_inv(1.0 - 1.0 / N)
            + euler_gamma * _phi_inv(1.0 - 1.0 / (N * math.e))
        )

    # Benchmark SR* at the scale of our bar count
    sr_benchmark = max(sr_star, e_max / math.sqrt(T_bar))

    # Use the best fold's Sharpe for the PSR computation
    best_idx = int(np.argmax(sharpe_values))
    sr_best = sharpe_values[best_idx]
    T_best = n_bars_values[best_idx] if n_bars_values else int(T_bar)

    # Approximate skew/kurt as 0 (Gaussian)
    return psr(sr_best, T_best, skew=0.0, excess_kurt=0.0, sr_star=sr_benchmark)


def _phi_inv(p: float) -> float:
    """Inverse normal CDF (probit)."""
    if _HAS_SCIPY:
        return float(_norm.ppf(p))
    # Rational approximation valid for 0 < p < 1
    p = max(1e-12, min(1.0 - 1e-12, p))
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    approx = t - (c[0] + c[1] * t + c[2] * t ** 2) / (
        1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3)
    return -approx if p < 0.5 else approx


# ---------------------------------------------------------------------------
# Walk-Forward Efficiency
# ---------------------------------------------------------------------------

def walk_forward_efficiency(fold_results: List[dict]) -> float:
    """
    WFE = mean(OOS Sharpe) / mean(IS Sharpe proxy).

    IS proxy: -stage1_val_loss converted to a positive "IS quality" score.
    OOS: strategy_sharpe per fold.

    WFE ≈ 1.0 → no degradation (no overfitting).
    WFE << 1.0 → OOS performance much worse than IS (overfit).
    WFE < 0   → OOS Sharpe is negative while IS looked good (severe overfit).
    """
    if not fold_results:
        return float("nan")

    oos_sharpes = [fr.get("strategy_sharpe", 0.0) for fr in fold_results]

    # IS quality: lower val_loss = better → use 1/(1+loss) as a bounded positive score
    is_losses = [fr.get("stage1_val_loss", 1.0) for fr in fold_results]
    is_quality = [1.0 / (1.0 + max(l, 0.0)) for l in is_losses]

    mean_oos = float(np.mean(oos_sharpes))
    mean_is = float(np.mean(is_quality))

    if abs(mean_is) < 1e-10:
        return float("nan")
    return round(mean_oos / mean_is, 4)


# ---------------------------------------------------------------------------
# CSCV-based PBO
# ---------------------------------------------------------------------------

def compute_pbo(fold_results: List[dict]) -> dict:
    """
    Probability of Backtest Overfitting via CSCV.

    Treats each fold model as a distinct "strategy variant".
    Generates all C(N, N//2) IS/OOS assignments of fold indices.

    For each assignment:
      IS set  : half the folds → select the one with best IS quality (lowest val_loss)
      OOS set : remaining folds → rank the IS-selected fold's OOS Sharpe among OOS folds
      ω       : logit( relative OOS rank ) — negative means below median → overfit

    PBO = fraction(ω < 0).

    Returns
    -------
    dict with keys:
      pbo            : float in [0, 1]  (0 = no overfitting, 1 = always overfit)
      omega_mean     : mean logit rank across all combinations
      omega_std      : std  logit rank
      n_combinations : number of CSCV combinations used
      wfe            : Walk-Forward Efficiency ratio
      psr_per_fold   : list of PSR(SR*=0) values per fold
      dsr            : Deflated Sharpe Ratio (scalar)
      interpretation : human-readable verdict
    """
    N = len(fold_results)
    if N < 2:
        return {
            "pbo": None,
            "omega_mean": None,
            "omega_std": None,
            "n_combinations": 0,
            "wfe": float("nan"),
            "psr_per_fold": [],
            "dsr": None,
            "interpretation": "Insufficient folds for PBO (need ≥ 2)",
        }

    # Metrics per fold
    is_quality = np.array([
        1.0 / (1.0 + max(fr.get("stage1_val_loss", 1.0), 0.0))
        for fr in fold_results
    ])
    oos_sharpe = np.array([fr.get("strategy_sharpe", 0.0) for fr in fold_results])
    n_bars_arr = np.array([fr.get("n_bars", 252) for fr in fold_results])

    half = N // 2
    all_combos = list(combinations(range(N), half))

    omega_values = []

    for is_idx in all_combos:
        is_set  = list(is_idx)
        oos_set = [i for i in range(N) if i not in is_set]

        # Best strategy in IS: fold with highest IS quality
        best_pos = int(np.argmax(is_quality[is_set]))
        best_fold = is_set[best_pos]

        # OOS rank of the IS-selected fold among all oos_set folds + itself
        oos_competitors = np.array([oos_sharpe[i] for i in oos_set] + [oos_sharpe[best_fold]])
        rank = int((oos_sharpe[best_fold] > np.array([oos_sharpe[i] for i in oos_set])).sum()) + 1
        n_oos_total = len(oos_set) + 1

        # ω = logit(relative rank), negative → below median → overfit
        rel = rank / (n_oos_total + 1)
        rel = max(1e-6, min(1.0 - 1e-6, rel))
        omega = math.log(rel / (1.0 - rel))
        omega_values.append(omega)

    omega_arr = np.array(omega_values)
    pbo_val = float((omega_arr < 0).mean())

    # PSR per fold (SR* = 0, approximate skew/kurt = 0)
    psr_per_fold = []
    for i, fr in enumerate(fold_results):
        sr = fr.get("strategy_sharpe", 0.0)
        T  = int(n_bars_arr[i])
        psr_per_fold.append(round(psr(sr, T, skew=0.0, excess_kurt=0.0, sr_star=0.0), 4))

    # DSR
    dsr_val = dsr(list(oos_sharpe), list(n_bars_arr.astype(int)))

    # WFE
    wfe_val = walk_forward_efficiency(fold_results)

    # Interpretation
    interpretation = _interpret(pbo_val, wfe_val, dsr_val, psr_per_fold)

    return {
        "pbo":            round(pbo_val, 4),
        "omega_mean":     round(float(omega_arr.mean()), 4),
        "omega_std":      round(float(omega_arr.std()),  4),
        "n_combinations": len(all_combos),
        "wfe":            wfe_val,
        "psr_per_fold":   psr_per_fold,
        "dsr":            round(dsr_val, 4),
        "interpretation": interpretation,
    }


def _interpret(pbo: float, wfe: float, dsr: float, psr_vals: list) -> str:
    flags = []

    if pbo is not None:
        if pbo > 0.55:
            flags.append(f"HIGH OVERFIT: PBO={pbo:.2f} (>0.55)")
        elif pbo > 0.40:
            flags.append(f"MODERATE OVERFIT: PBO={pbo:.2f}")
        else:
            flags.append(f"LOW OVERFIT: PBO={pbo:.2f} (<0.40)")

    if not math.isnan(wfe):
        if wfe < 0:
            flags.append(f"OOS Sharpe negative (WFE={wfe:.2f})")
        elif wfe < 0.5:
            flags.append(f"Large IS-to-OOS decay (WFE={wfe:.2f})")
        else:
            flags.append(f"Acceptable IS/OOS ratio (WFE={wfe:.2f})")

    if dsr is not None:
        if dsr < 0.5:
            flags.append(f"Sharpe not significant after multi-test correction (DSR={dsr:.2f})")
        else:
            flags.append(f"Sharpe survives multiple testing (DSR={dsr:.2f})")

    if psr_vals:
        mean_psr = float(np.mean(psr_vals))
        flags.append(f"Mean PSR(SR*=0)={mean_psr:.2f}")

    return " | ".join(flags) if flags else "OK"


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_pbo_report(result: dict, phase_name: str = "") -> None:
    label = f"  PBO Report{(' - ' + phase_name) if phase_name else ''}"
    print(f"\n{label}")
    print("  " + "-" * (len(label) - 2))

    if result.get("pbo") is None:
        print(f"  {result['interpretation']}")
        return

    print(f"  PBO (CSCV):      {result['pbo']:.4f}  "
          f"({result['n_combinations']} combinations)")
    print(f"  Omega:           mean={result['omega_mean']:+.4f}  "
          f"std={result['omega_std']:.4f}")
    print(f"  Walk-Fwd Eff:    {result['wfe']:.4f}"
          f"  (OOS/IS Sharpe ratio; 1.0 = ideal)")
    if result['dsr'] is not None:
        print(f"  DSR:             {result['dsr']:.4f}"
              f"  (Deflated Sharpe Ratio)")
    if result['psr_per_fold']:
        psr_str = "  ".join(f"F{i}={v:.2f}" for i, v in enumerate(result['psr_per_fold']))
        print(f"  PSR per fold:    {psr_str}")
    print(f"  Verdict:         {result['interpretation']}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate 4 folds: IS val_loss and OOS Sharpe
    fold_results = [
        {"fold": 0, "stage1_val_loss": 0.012, "strategy_sharpe":  0.45, "n_bars": 500, "accuracy": 0.55},
        {"fold": 1, "stage1_val_loss": 0.010, "strategy_sharpe":  0.30, "n_bars": 500, "accuracy": 0.53},
        {"fold": 2, "stage1_val_loss": 0.009, "strategy_sharpe": -0.10, "n_bars": 500, "accuracy": 0.48},
        {"fold": 3, "stage1_val_loss": 0.008, "strategy_sharpe":  0.60, "n_bars": 500, "accuracy": 0.57},
    ]

    result = compute_pbo(fold_results)
    print_pbo_report(result, phase_name="smoke_test")
    print(f"\n  Raw result: {result}")
