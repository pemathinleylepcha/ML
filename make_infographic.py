import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(20, 11.25))
fig.patch.set_facecolor('#0B1120')
ax.set_facecolor('#0B1120')
ax.set_xlim(0, 20)
ax.set_ylim(0, 11.25)
ax.axis('off')

def box(x, y, w, h, fc, ec=None, lw=2.0, radius=0.20, zorder=3):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle=f'round,pad=0.0,rounding_size={radius}',
        facecolor=fc, edgecolor=ec if ec else 'none',
        linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, size=9, color='white', weight='normal', ha='center', va='center', zorder=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=zorder, fontfamily='monospace')

def arr(x1, y1, x2, y2, color='#475569', lw=2.0, style='arc3,rad=0.0', bidir=False, zorder=2):
    astyle = '<->' if bidir else '->'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=astyle, color=color, lw=lw,
                        connectionstyle=style), zorder=zorder)

# ── column centres ─────────────────────────────────────────────────────────
C1, C2, C3, C4, C5 = 1.80, 5.10, 10.0, 14.8, 18.5

# ── TITLE ──────────────────────────────────────────────────────────────────
txt(10, 10.65, 'ALGO C2 v2  —  Dual-Subnet Signal Architecture', 17, '#E2E8F0', 'bold')
txt(10, 10.18, '43-Node Graph  ·  CatBoost-BTC (24x7)  ·  CatBoost-FX (24x5)  ·  Walk-Forward + PBO', 9.5, '#64748B')

# ── COLUMN HEADERS ─────────────────────────────────────────────────────────
for cx, label in [(C1,'DATA'), (C2,'GRAPH ENGINE'), (C3,'DUAL SUBNET + BRIDGE'), (C4,'CATBOOST'), (C5,'EXECUTION')]:
    box(cx, 9.50, 3.0, 0.42, '#1E293B', radius=0.12, zorder=2)
    txt(cx, 9.50, label, 8, '#94A3B8', 'bold')

# divider lines
for xd in [3.47, 6.65, 13.35, 16.65]:
    ax.plot([xd, xd], [0.85, 9.28], color='#1E293B', lw=1.2, zorder=1)

# ── DATA (C1) ──────────────────────────────────────────────────────────────
box(C1, 8.30, 3.0, 0.95, '#0F2A45', '#1D4ED8', 1.5, zorder=3)
txt(C1, 8.57, '29 Tradeable Instruments', 9, '#60A5FA', 'bold')
txt(C1, 8.30, 'BTCUSD (24x7)  +  28 FX Pairs', 8, '#93C5FD')
txt(C1, 8.05, 'Majors · Minors · Crosses', 7.5, '#BFDBFE')

box(C1, 7.10, 3.0, 0.75, '#1A0F45', '#7C3AED', 1.5, zorder=3)
txt(C1, 7.32, '14 Signal-Only Nodes', 9, '#A78BFA', 'bold')
txt(C1, 7.07, 'Indices · Metals · Energy · Exotic FX', 7.5, '#C4B5FD')

box(C1, 6.25, 3.0, 0.65, '#0F2A1A', '#059669', 1.5, zorder=3)
txt(C1, 6.45, 'Historical 2019–2026', 9, '#34D399', 'bold')
txt(C1, 6.22, 'M1 / M5 / H1  (Progressive)', 7.5, '#6EE7B7')

box(C1, 5.47, 3.0, 0.60, '#2A0F1A', '#BE185D', 1.5, zorder=3)
txt(C1, 5.67, 'Live MT5 Tick Stream', 9, '#F472B6', 'bold')
txt(C1, 5.44, '1000ms Microstructure Bars', 7.5, '#FBCFE8')

# ── GRAPH ENGINE (C2) ──────────────────────────────────────────────────────
box(C2, 8.28, 3.1, 0.98, '#0F1E3A', '#1D4ED8', 1.5, zorder=3)
txt(C2, 8.55, 'Mantegna Correlation Graph', 9, '#60A5FA', 'bold')
txt(C2, 8.28, 'Correlation -> Distance -> Laplacian', 7.5, '#93C5FD')
txt(C2, 8.03, '43 x 43 spectral decomposition', 7, '#BFDBFE')

box(C2, 7.10, 3.1, 0.80, '#1A0F3A', '#7C3AED', 1.5, zorder=3)
txt(C2, 7.32, 'TDA Topology', 9, '#818CF8', 'bold')
txt(C2, 7.07, 'Beta-0  Beta-1  H1-lifespan  Lambda-2', 7.5, '#A5B4FC')

box(C2, 6.24, 3.1, 0.65, '#0F2A1A', '#059669', 1.5, zorder=3)
txt(C2, 6.44, 'Regime Classifier', 9, '#34D399', 'bold')
txt(C2, 6.21, 'LOW_VOL  NORMAL  HIGH_STRESS  FRAG', 7, '#6EE7B7')

box(C2, 5.50, 3.1, 0.65, '#2A1A0F', '#D97706', 1.5, zorder=3)
txt(C2, 5.70, 'Node Residuals', 9, '#FCD34D', 'bold')
txt(C2, 5.47, 'Laplacian per-node residuals  (43)', 7.5, '#FDE68A')

# ── BTC SUBNET (C3 upper) ──────────────────────────────────────────────────
box(C3, 8.38, 6.5, 1.55, '#1A0D00', '#F59E0B', 2.2, zorder=3)
txt(C3, 8.93, 'BTC Subnet  24x7', 12, '#FBBF24', 'bold')
txt(C3, 8.62, '37 features per bar', 9, '#FDE68A')
txt(C3, 8.38, 'Technical (16)  Microstructure (4)', 8, '#FEF3C7')
txt(C3, 8.16, 'Laplacian residual (1)  TDA (4)', 8, '#FEF3C7')
txt(C3, 7.94, 'Regime (1)  FX->BTC bridge (6)', 8, '#FEF3C7')
txt(C3, 7.72, 'Shared graph (5)', 8, '#FEF3C7')

# ── BRIDGE BAND (C3 centre) ────────────────────────────────────────────────
box(C3, 6.88, 6.5, 0.72, '#1A0A2A', '#9333EA', 2.0, zorder=4)
txt(C3, 7.10, 'CROSS-LEARNING BRIDGE', 8, '#C084FC', 'bold')
txt(C3, 6.88, 'BTC->FX   8 features', 7.5, '#DDD6FE')
txt(C3, 6.67, 'FX->BTC   6 features', 7.5, '#DDD6FE')
txt(C3, 6.45, 'Shared    5 features', 7.5, '#E9D5FF')

# ── FX SUBNET (C3 lower) ───────────────────────────────────────────────────
box(C3, 5.38, 6.5, 1.55, '#001228', '#3B82F6', 2.2, zorder=3)
txt(C3, 5.93, 'FX Subnet  24x5', 12, '#60A5FA', 'bold')
txt(C3, 5.62, '138 features per pair', 9, '#BAE6FD')
txt(C3, 5.38, 'Technical (16)  Micro (4)  Residual (2)', 8, '#E0F2FE')
txt(C3, 5.16, 'Signal-only block  98  (14 x 7)', 8, '#E0F2FE')
txt(C3, 4.94, 'TDA (5)  BTC->FX bridge (8)  Shared (5)', 8, '#E0F2FE')

# ── CATBOOST (C4 upper) ────────────────────────────────────────────────────
box(C4, 8.38, 3.1, 1.55, '#1A1000', '#F59E0B', 2.2, zorder=3)
txt(C4, 8.91, 'CatBoost-BTC', 12, '#FBBF24', 'bold')
txt(C4, 8.62, 'Input: 37 features', 8.5, '#FDE68A')
txt(C4, 8.38, 'Output: P(BUY) P(HOLD) P(SELL)', 8, '#FEF3C7')
txt(C4, 8.15, 'Depth 6  Iter 500  LR 0.05', 7.5, '#FDE68A')
txt(C4, 7.92, 'Walk-Fwd CV 5 folds + PBO', 7.5, '#FDE68A')

box(C4, 5.38, 3.1, 1.55, '#001020', '#3B82F6', 2.2, zorder=3)
txt(C4, 5.91, 'CatBoost-FX', 12, '#60A5FA', 'bold')
txt(C4, 5.62, 'Input: 138 + pair_id (categorical)', 8.5, '#BAE6FD')
txt(C4, 5.38, 'Output: P(BUY) P(HOLD) P(SELL)', 8, '#E0F2FE')
txt(C4, 5.15, '28 pairs pooled  -  1 shared model', 7.5, '#BAE6FD')
txt(C4, 4.92, 'Walk-Fwd CV 5 folds + PBO', 7.5, '#BAE6FD')

# ── EXECUTION (C5) ─────────────────────────────────────────────────────────
box(C5, 8.38, 2.7, 1.55, '#001808', '#22C55E', 2.2, zorder=3)
txt(C5, 8.88, '6-Gate Filter', 11, '#4ADE80', 'bold')
txt(C5, 8.60, 'G1 Net Score   G2 CB Floor', 7.5, '#86EFAC')
txt(C5, 8.38, 'G3 EMA Trend   G4 ATR Band', 7.5, '#86EFAC')
txt(C5, 8.16, 'G5 CB Agree    G6 Regime', 7.5, '#86EFAC')
txt(C5, 7.94, 'All 6 must pass', 7.5, '#4ADE80', 'bold')

box(C5, 6.80, 2.7, 0.90, '#001808', '#22C55E', 1.8, zorder=3)
txt(C5, 7.02, 'Position Sizing', 9, '#4ADE80', 'bold')
txt(C5, 6.79, 'ATR-based SL / TP  2% risk', 7.5, '#86EFAC')
txt(C5, 6.56, 'Leverage 500  $50 account', 7.5, '#86EFAC')

box(C5, 5.82, 2.7, 0.72, '#200010', '#F43F5E', 2.0, zorder=3)
txt(C5, 6.02, 'MT5  Order Placement', 9, '#FB7185', 'bold')
txt(C5, 5.79, 'Tradeable guard  Session filter', 7.5, '#FECDD3')

# ── PBO PANEL ──────────────────────────────────────────────────────────────
box(14.8, 3.78, 7.8, 1.60, '#040D1A', '#22C55E', 1.5, zorder=3)
txt(14.8, 4.33, 'PBO  Analysis  (Bailey et al. 2014)', 9.5, '#4ADE80', 'bold')
for i, (lbl, val) in enumerate([('PBO (CSCV)', '<0.40 target'), ('DSR', 'multi-test corrected'),
                                  ('PSR', 'P(SR > 0) per fold'), ('WFE', 'OOS/IS Sharpe ~ 1.0')]):
    cx = 11.5 + i * 2.15
    txt(cx, 4.00, lbl, 8, '#86EFAC', 'bold')
    txt(cx, 3.77, val, 7, '#4B5563')
txt(14.8, 3.53, 'CSCV generates C(N, N/2) IS/OOS splits  -  fraction(omega < 0) = PBO', 7.5, '#374151')

# ── PROGRESSIVE TRAINING ───────────────────────────────────────────────────
box(7.55, 2.77, 15.1, 1.60, '#0A0A1E', '#7C3AED', 1.5, zorder=3)
txt(7.55, 3.33, 'Progressive Training Pipeline', 9.5, '#A78BFA', 'bold')
phases = [
    ('Phase 1', '2019-2020', 'H1 Macro', '#60A5FA'),
    ('Phase 2', '2021-2023', 'H1 Intraday', '#818CF8'),
    ('Phase 3', '2024-2025', 'M5 Recent', '#C084FC'),
    ('Phase 3b', '2025-2026', 'M1 Micro', '#F472B6'),
]
for i, (ph, yr, tf, col) in enumerate(phases):
    cx = 2.0 + i * 2.5
    box(cx, 3.0 - 0.44, 2.2, 0.72, '#14103A', radius=0.14, zorder=4)
    txt(cx, 2.87, ph, 9, col, 'bold')
    txt(cx, 2.65, yr, 7.5, '#CBD5E1')
    txt(cx, 2.43, tf, 7.5, col)
for i in range(3):
    arr(2.0 + i*2.5 + 1.12, 2.65, 2.0 + (i+1)*2.5 - 1.12, 2.65, '#4C1D95', 2.0)

# ── ARROWS ─────────────────────────────────────────────────────────────────
# data -> engine
for y in [8.28, 7.10, 6.24, 5.50]:
    arr(C1+1.52, y, C2-1.57, y, '#334155', 1.8)

# engine -> subnets
arr(C2+1.57, 8.0, C3-3.27, 8.15, '#1D4ED8', 2.0)
arr(C2+1.57, 5.8, C3-3.27, 5.60, '#1D4ED8', 2.0)

# subnets -> catboost
arr(C3+3.27, 8.38, C4-1.57, 8.38, '#F59E0B', 2.5)
arr(C3+3.27, 5.38, C4-1.57, 5.38, '#3B82F6', 2.5)

# catboost -> gates
arr(C4+1.57, 8.38, C5-1.37, 8.38, '#F59E0B', 2.0)
arr(C4+1.57, 5.38, C5-1.37, 6.45, '#3B82F6', 2.0, 'arc3,rad=-0.25')

# gates -> sizing -> mt5
arr(C5, 7.60, C5, 7.26, '#22C55E', 2.0)
arr(C5, 6.35, C5, 6.19, '#22C55E', 2.0)

# bridge arrows (bidirectional)
arr(C3-2.05, 8.05, C3-1.10, 7.15, '#9333EA', 1.8, 'arc3,rad=-0.3', True, 5)
arr(C3-2.05, 5.70, C3-1.10, 6.60, '#9333EA', 1.8, 'arc3,rad=0.3', True, 5)

# ── BOTTOM STATS ───────────────────────────────────────────────────────────
box(10.0, 1.35, 19.5, 1.22, '#111827', radius=0.18, zorder=2)
stats = [
    ('43 Nodes', '29 tradeable + 14 signal-only', '#60A5FA'),
    ('570 Features', '16x29 + 7x14 + 8 graph', '#A78BFA'),
    ('37 / 139', 'BTC / FX features per bar', '#FBBF24'),
    ('CSCV PBO', 'Overfitting guard pre-deploy', '#4ADE80'),
    ('Apr 13 Demo', 'Apr 20 Live  $50 capital', '#F472B6'),
]
for i, (title, sub, col) in enumerate(stats):
    cx = 1.5 + i * 4.25
    txt(cx, 1.58, title, 10, col, 'bold')
    txt(cx, 1.25, sub, 7.5, '#4B5563')

plt.tight_layout(pad=0)
out = 'd:/Algo-C2/algo_c2_v2_architecture.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#0B1120', edgecolor='none')
print(f'Saved: {out}')
