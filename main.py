#!/usr/bin/env python
"""Main standalone entrypoint for TOG-main Polymarket Backtester & Strategy Optimizer.

Run via:
    python Scripts/main.py
or:
    cd Scripts && python main.py
"""

import os
import sys
import json
from pathlib import Path

# Add current folder to sys.path so it works standalone
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_data import load_config, prepare_dataset
from backtest import (
    run_simulation,
    export_results,
    run_optimization_suite,
    export_optimization_report
)


def print_banner(optimization_mode: bool):
    mode_text = "OPTIMIZATION MODE (32 Configurations)" if optimization_mode else "SINGLE BACKTEST MODE"
    banner = f"""
================================================================================
          TOG-MAIN POLYMARKET BTC 15M QUANTITATIVE ENGINE
                      [{mode_text}]
================================================================================
    """
    print(banner)


def display_single_results(results: dict):
    if "error" in results:
        print(f"\n[!] BACKTEST ERROR: {results['error']}")
        return

    s = results.get("summary", {})
    bs = results.get("breakdown_side", {})

    print("\n" + "=" * 80)
    print("                           FULL BACKTEST RESULTS")
    print("=" * 80)
    print(f" Strategy Mode        : {s.get('mode_label')}")
    print(f" Initial Capital      : ${s.get('initial_balance'):,.2f}")
    print(f" Final Capital        : ${s.get('final_balance'):,.2f}")
    print(f" Net Profit           : ${s.get('net_profit'):+,.2f} ({s.get('total_return_pct'):+.2f}%)")
    print(f" Peak Capital         : ${s.get('peak_balance'):,.2f}")
    print(f" Max Drawdown         : -${s.get('max_drawdown'):,.2f} (-{s.get('max_drawdown_pct'):.2f}%)")
    print("-" * 80)
    print(f" Total Markets        : {s.get('markets_evaluated'):,}")
    print(f" Executed Trades      : {s.get('total_trades'):,} ({s.get('trade_frequency_pct')}% market participation)")
    print(f" Vetoed Signals       : {s.get('veto_count'):,}")
    print(f" Win Rate             : {s.get('win_rate')}% ({s.get('winning_trades')} Wins / {s.get('losing_trades')} Losses)")
    print(f" Profit Factor        : {s.get('profit_factor')}")
    print(f" Avg Win / Avg Loss   : +${s.get('avg_win'):.2f} / ${s.get('avg_loss'):.2f} (Ratio: {s.get('win_loss_ratio')})")
    print(f" Total Taker Fees     : ${s.get('total_fees_paid'):.2f}")
    print(f" Annualized Sharpe    : {s.get('sharpe_ratio')}")
    print("-" * 80)
    print(" DIRECTIONAL PERFORMANCE:")
    print(f"   UP   Trades : {bs.get('UP', {}).get('trades'):5d} | Win Rate: {bs.get('UP', {}).get('win_rate'):5.1f}% | PnL: ${bs.get('UP', {}).get('pnl'):+10.2f}")
    print(f"   DOWN Trades : {bs.get('DOWN', {}).get('trades'):5d} | Win Rate: {bs.get('DOWN', {}).get('win_rate'):5.1f}% | PnL: ${bs.get('DOWN', {}).get('pnl'):+10.2f}")
    print("=" * 80)


def display_optimization_summary(suite_results: dict):
    leaderboard = suite_results.get("leaderboard", [])
    best = suite_results.get("best_strategy", {})
    reg = suite_results.get("regime_diagnostics", {})

    print("\n" + "=" * 90)
    print("                       TOP 10 OPTIMIZATION LEADERBOARD")
    print("=" * 90)
    print(f"{'Rank':<5} {'Strategy Name':<42} {'Vetoes':<8} {'Win Rate':<10} {'Trades':<8} {'Net PnL ($)':<12}")
    print("-" * 90)

    for rank, s in enumerate(leaderboard[:10], 1):
        veto_str = "ON" if s["use_vetoes"] else "OFF"
        print(f"#{rank:<4} {s['name']:<42} {veto_str:<8} {s['win_rate']:<5.1f}%    {s['trades']:<8} ${s['net_pnl']:+,.2f}")

    print("=" * 90)
    print("\n" + "*" * 90)
    print(f"  #1 BEST PERFORMING STRATEGY SPOTLIGHT: {best.get('name')}")
    print("*" * 90)
    print(f"  Category       : {best.get('category')}")
    print(f"  Win Rate       : {best.get('win_rate'):.2f}%")
    print(f"  Net Profit     : ${best.get('net_pnl'):+,.2f} ({best.get('total_return_pct'):+.2f}%)")
    print(f"  Profit Factor  : {best.get('profit_factor'):.2f}")
    print(f"  Executed Trades: {best.get('trades'):,} ({best.get('trade_frequency_pct'):.1f}% of markets)")
    print(f"  Max Drawdown   : -${best.get('max_drawdown'):,.2f} (-{best.get('max_drawdown_pct'):.2f}%)")
    print(f"  Avg Win / Loss : +${best.get('avg_win'):.2f} / ${best.get('avg_loss'):.2f} (Ratio: {best.get('win_loss_ratio'):.2f})")
    print("-" * 90)
    print("  Recommended config settings to deploy this strategy:")
    print("  " + json.dumps({
        "strategy": {
            "ev_threshold": best.get("params", {}).get("ev_threshold", 0.04),
            "min_prob": best.get("params", {}).get("min_prob", 0.55),
            "min_strike_distance_bp": best.get("params", {}).get("min_strike_distance_bp", 0.0),
            "min_entry_second": best.get("params", {}).get("min_entry_second", 0),
            "max_entry_second": best.get("params", {}).get("max_entry_second", 840),
            "use_15m_ha": best.get("params", {}).get("use_15m_ha", False),
            "use_15m_trend_ema": best.get("params", {}).get("use_15m_trend_ema", False),
            "net_ev_gate": best.get("params", {}).get("net_ev_gate", False)
        },
        "use_vetoes": best.get("use_vetoes", False)
    }, indent=4).replace("\n", "\n  "))
    print("*" * 90)


def main():
    cfg_path = CURRENT_DIR / "config.json"
    if not cfg_path.exists():
        print(f"[!] config.json not found at {cfg_path}")
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    optimization_mode = bool(cfg.get("optimization_mode", False))
    print_banner(optimization_mode)

    output_dir = CURRENT_DIR / cfg.get("output_dir", "results")

    # Step 1: Prepare/Fetch Data
    try:
        data_file = prepare_dataset(cfg)
    except Exception as e:
        print(f"\n[!] Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Run Backtest or Optimization Suite
    if optimization_mode:
        try:
            suite_results = run_optimization_suite(data_file, cfg)
        except Exception as e:
            print(f"\n[!] Optimization suite error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Step 3: Export and Display Optimization Results
        export_optimization_report(suite_results, str(output_dir))
        display_optimization_summary(suite_results)

        print(f"\n[OK] Optimization files exported to: {output_dir}")
        print(f"     - Unified Optimization Report : {output_dir / 'optimization_report.md'}")
        print(f"     - Ranked Leaderboard (CSV)    : {output_dir / 'optimization_leaderboard.csv'}")
        print(f"     - Champion Strategy Trades    : {output_dir / 'best_strategy_trades.csv'}\n")

    else:
        try:
            results = run_simulation(data_file, cfg)
        except Exception as e:
            print(f"\n[!] Simulation error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Step 3: Export and Display Single Results
        export_results(results, str(output_dir))
        display_single_results(results)

        print(f"\n[OK] Full backtest files exported to: {output_dir}")
        print(f"     - Summary Report : {output_dir / 'summary_report.md'}")
        print(f"     - Detailed Trades: {output_dir / 'trades.csv'}")
        print(f"     - Equity Curve   : {output_dir / 'equity_curve.csv'}\n")


if __name__ == "__main__":
    main()
