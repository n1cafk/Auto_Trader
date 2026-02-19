"""Streamlit dashboard for paper-trading bot monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.settings import AppSettings, load_settings
from src.dashboard.data_loader import (
    available_symbols,
    load_equity_curve,
    load_json,
    load_trade_journal,
    read_recent_log_lines,
    summarize_from_artifacts,
)

st.set_page_config(page_title="Auto Trader Dashboard", layout="wide")


def _symbol_choices(settings: AppSettings, track: str) -> list[str]:
    discovered = available_symbols(settings.reports_dir, track)
    configured = sorted({*settings.safe_live_symbol_list, *settings.training_symbol_list})
    return sorted({*discovered, *configured})


def _show_status(settings: AppSettings, risk_limits: object) -> None:
    st.subheader("Runtime Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", settings.mode.value.upper())
    c2.metric("Market", settings.market)
    c3.metric("Data Provider", settings.data_provider)
    c4.metric("Execution Provider", settings.execution_provider)
    st.caption(
        f"Long-term timeframe: {settings.long_term_timeframe} | "
        f"Intraday timeframe: {settings.intraday_timeframe} | "
        f"Report TZ: {settings.report_timezone}",
    )
    st.json(risk_limits.model_dump())


def _show_backtest(track: str, symbol: str, report_root: Path) -> None:
    st.subheader("Backtest Summary")
    report_path = report_root / "backtest_report.json"
    report = load_json(report_path)
    if not report:
        st.info("No backtest report found yet for selected track/symbol.")
        return
    gate = report.get("promotion_gate", {})
    metrics = report.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Gate Decision", gate.get("decision", "unknown"))
    c2.metric("Sharpe", f'{metrics.get("sharpe", 0):.3f}')
    c3.metric("Max Drawdown", f'{metrics.get("max_drawdown", 0):.3%}')
    st.json(report)


def _show_paper_results(track: str, symbol: str, paper_root: Path) -> None:
    st.subheader("Paper Trading Results")
    equity_path = paper_root / "paper_equity_curve.csv"
    trades_path = paper_root / "paper_trades.csv"
    equity_df = load_equity_curve(equity_path)
    trades_df = load_trade_journal(trades_path)

    if equity_df.empty:
        st.info("No paper equity curve found yet.")
        return

    metrics = summarize_from_artifacts(equity_path, trades_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics.total_return:.2%}")
    c2.metric("Sharpe", f"{metrics.sharpe:.3f}")
    c3.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    c4.metric("Trade Count", f"{metrics.trade_count}")

    time_col = "timestamp_report_tz" if "timestamp_report_tz" in equity_df.columns else "timestamp_utc"
    if time_col not in equity_df.columns:
        time_col = "timestamp"
    plot_df = equity_df.copy()
    if time_col in plot_df.columns:
        plot_df = plot_df.dropna(subset=[time_col])
        if not plot_df.empty:
            fig = px.line(plot_df, x=time_col, y="equity", title=f"{symbol} {track} Equity Curve")
            st.plotly_chart(fig, use_container_width=True)

    if not trades_df.empty:
        st.markdown("#### Recent Trades")
        show_cols = [c for c in ["timestamp", "symbol", "side", "fill_price", "quantity", "fee", "realized_pnl"] if c in trades_df.columns]
        st.dataframe(trades_df[show_cols].tail(30), use_container_width=True)
    else:
        st.caption("No trades recorded yet.")


def _show_models(settings: AppSettings, track: str) -> None:
    st.subheader("Model Metadata")
    metadata_path = settings.metadata_path_for_track(track)
    metadata = load_json(metadata_path)
    if not metadata:
        st.info(f"Metadata not found at {metadata_path}.")
        return
    st.json(metadata)


def _show_logs(settings: AppSettings) -> None:
    st.subheader("Recent Logs")
    lines = read_recent_log_lines(settings.logs_dir / "paper_runner.log", max_lines=120)
    if not lines:
        st.info("No paper runner logs yet.")
        return
    st.code("\n".join(lines), language="json")


def main() -> None:
    settings, risk_limits = load_settings()
    st.title("Auto Trader Dashboard")
    st.caption("Paper-first monitoring for US ETFs/stocks workflow")

    track = st.sidebar.selectbox("Track", options=["long_term", "intraday"], index=0)
    symbols = _symbol_choices(settings, track)
    if not symbols:
        st.warning("No symbols available. Generate reports first via CLI.")
        return
    symbol = st.sidebar.selectbox("Symbol", options=symbols, index=0)

    _show_status(settings, risk_limits)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        _show_backtest(track, symbol, settings.reports_dir / "backtest" / track / symbol)
    with col_right:
        _show_models(settings, track)

    st.divider()
    _show_paper_results(track, symbol, settings.reports_dir / "paper" / track / symbol)
    st.divider()
    _show_logs(settings)


if __name__ == "__main__":
    main()
