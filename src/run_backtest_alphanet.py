"""Run backtest with AlphaNet model on remote machine."""
import json
import torch
from backtester import Backtester, BacktestConfig
from market_neutral_model import AlphaNet

print("Loading data...")
with open("C:/Algo-C2/data/algo_c2_5day_data.json") as f:
    data = json.load(f)

print("Loading model...")
ckpt = torch.load("C:/Algo-C2/models/alpha_net.pt", map_location="cpu", weights_only=False)
model = AlphaNet(n_features=ckpt["n_features"], n_pairs=ckpt["n_pairs"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# cb_thr=0.72: when using AlphaNet, G2 checks max(cb_scores) which is the most-confident
# pair's sigmoid confidence. max_cb ranges 0.62-0.82 (mean=0.70), so 0.72 gives ~40% pass rate.
config = BacktestConfig(init_balance=50.0, cb_thr=0.72, risk_pct=0.02, use_session_filter=False)
bt = Backtester(
    config,
    alpha_model=model,
    alpha_scaler=(ckpt["scaler_mean"], ckpt["scaler_std"]),
    alpha_pairs=ckpt["pairs"],
)
results = bt.run(data)
m = results.metrics
gs = results.gate_stats

print("--- v2 + AlphaNet (fixed features) ---")
print(f"Trades     : {m['total_trades']}  ({m['total_trades']/18:.1f}/day)")
print(f"Win rate   : {m['win_rate']:.1%}")
print(f"Avg win    : ${m['avg_win']:.4f}  |  Avg loss: ${m['avg_loss']:.4f}")
print(f"EV/trade   : ${m['ev']:.4f}")
print(f"Profit fac : {m['profit_factor']:.2f}")
print(f"Max DD     : {m['max_drawdown']:.2%}")
print(f"Return     : {m['return_pct']:.2%}")
print(f"Spread drag: ${m['spread_drag']:.4f}")
print(f"Sharpe     : {m['sharpe']:.2f}")
print(f"Final bal  : ${m['final_balance']:.2f}")
print(f"Gates      : {gs}")

if results.trades:
    by_pair = {}
    for t in results.trades:
        by_pair.setdefault(t["pair"], []).append(t["pnl_usd"])
    top = sorted(by_pair.items(), key=lambda x: sum(x[1]), reverse=True)[:5]
    print("Top pairs by PnL:")
    for pair, pnls in top:
        print(f"  {pair}: {len(pnls)} trades, net=${sum(pnls):.4f}")

bt.export_results(results, "C:/Algo-C2/data/backtest_alphanet.json")
print("Done.")
