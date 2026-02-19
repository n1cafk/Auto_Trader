"""Command-line interface for AI paper-trading bot workflow."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config.settings import AppSettings, load_settings, mode_banner
from src.data.fetch_ohlcv import FetchRequest, fetch_ohlcv, generate_synthetic_ohlcv, load_ohlcv_csv, save_ohlcv_csv
from src.evaluation.backtest import run_backtest
from src.evaluation.gates import GateThresholds, evaluate_promotion
from src.evaluation.metrics import summarize_performance
from src.execution.runner import run_paper_session
from src.models.train import train_from_candles
from src.monitoring.logger import configure_logger, log_event


def _symbol_file(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_')}_{timeframe}.csv"


def _ensure_candles_for_symbol(settings: AppSettings, symbol: str) -> pd.DataFrame:
    data_path = settings.data_dir / _symbol_file(symbol, settings.timeframe)
    if not data_path.exists():
        candles = generate_synthetic_ohlcv(length=settings.data_limit)
        save_ohlcv_csv(candles, data_path)
    return load_ohlcv_csv(data_path)


def _ensure_model_exists(settings: AppSettings, candles: pd.DataFrame) -> None:
    if settings.model_path.exists() and settings.metadata_path.exists():
        return
    train_from_candles(
        candles=candles,
        model_path=settings.model_path,
        metadata_path=settings.metadata_path,
        horizon=settings.prediction_horizon,
        target_return_threshold=settings.target_return_threshold,
    )


def fetch_data(args: argparse.Namespace) -> None:
    settings, _ = load_settings()
    symbols = settings.symbol_list if not args.symbol else [args.symbol]
    for symbol in symbols:
        if args.synthetic:
            candles = generate_synthetic_ohlcv(length=settings.data_limit)
        else:
            request = FetchRequest(
                exchange_id=settings.exchange_name,
                symbol=symbol,
                timeframe=settings.timeframe,
                limit=settings.data_limit,
            )
            candles = fetch_ohlcv(request)
        out_path = settings.data_dir / _symbol_file(symbol, settings.timeframe)
        save_ohlcv_csv(candles, out_path)
        print(f"Saved candles: {out_path} ({len(candles)} rows)")


def train_model(args: argparse.Namespace) -> None:
    settings, _ = load_settings()
    symbol = args.symbol or settings.symbol_list[0]
    candles = _ensure_candles_for_symbol(settings, symbol)
    result = train_from_candles(
        candles=candles,
        model_path=settings.model_path,
        metadata_path=settings.metadata_path,
        horizon=settings.prediction_horizon,
        target_return_threshold=settings.target_return_threshold,
    )
    print("Model trained and saved.")
    print(json.dumps(result.validation_metrics, indent=2, sort_keys=True))


def backtest(args: argparse.Namespace) -> None:
    settings, risk_limits = load_settings()
    symbol = args.symbol or settings.symbol_list[0]
    candles = _ensure_candles_for_symbol(settings, symbol)
    _ensure_model_exists(settings, candles)
    report = run_backtest(
        candles=candles,
        symbol=symbol,
        settings=settings,
        risk_limits=risk_limits,
        report_dir=settings.reports_dir / "backtest",
    )
    print(f"Backtest report: {report.report_path}")
    print(f"Decision: {report.gate_decision}")
    for reason in report.gate_reasons:
        print(f"- {reason}")


def run_paper(args: argparse.Namespace) -> None:
    settings, risk_limits = load_settings()
    symbol = args.symbol or settings.symbol_list[0]
    log_file = settings.logs_dir / "paper_runner.log"
    logger = configure_logger(log_file)
    print(mode_banner(settings.mode))

    candles = _ensure_candles_for_symbol(settings, symbol)
    _ensure_model_exists(settings, candles)

    result = run_paper_session(
        candles=candles,
        symbol=symbol,
        settings=settings,
        risk_limits=risk_limits,
        output_dir=settings.reports_dir / "paper",
        logger=logger,
    )
    metrics = summarize_performance(result.equity_curve, result.trade_pnls)
    gate = evaluate_promotion(metrics, GateThresholds())
    log_event(
        logger,
        "paper_run_complete",
        {
            "symbol": symbol,
            "final_equity": result.final_equity,
            "decision": gate.decision.value,
        },
    )
    print(f"Paper trades: {result.trades_path}")
    print(f"Equity curve: {result.equity_path}")
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    print(f"Promotion decision: {gate.decision.value}")


def status(_: argparse.Namespace) -> None:
    settings, risk_limits = load_settings()
    print(mode_banner(settings.mode))
    print(f"Exchange: {settings.exchange_name}")
    print(f"Symbols: {', '.join(settings.symbol_list)}")
    print(f"Risk limits: {risk_limits.model_dump()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI auto trader (paper-first) CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-data", help="Fetch market candles")
    fetch.add_argument("--symbol", type=str, default=None, help="Override symbol")
    fetch.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic candles (offline testing).",
    )
    fetch.set_defaults(func=fetch_data)

    train = sub.add_parser("train", help="Train baseline model")
    train.add_argument("--symbol", type=str, default=None, help="Override symbol")
    train.set_defaults(func=train_model)

    bt = sub.add_parser("backtest", help="Run backtest + gate evaluation")
    bt.add_argument("--symbol", type=str, default=None, help="Override symbol")
    bt.set_defaults(func=backtest)

    paper = sub.add_parser("paper-run", help="Run paper trading simulation loop")
    paper.add_argument("--symbol", type=str, default=None, help="Override symbol")
    paper.set_defaults(func=run_paper)

    st = sub.add_parser("status", help="Show runtime config safety status")
    st.set_defaults(func=status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
