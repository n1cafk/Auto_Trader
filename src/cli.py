"""Command-line interface for AI paper-trading bot workflow."""

from __future__ import annotations

import argparse
import json
from typing import Literal

import pandas as pd

from src.config.settings import AppSettings, load_settings, mode_banner
from src.data.fetch_ohlcv import FetchRequest, fetch_ohlcv, generate_synthetic_ohlcv, load_ohlcv_csv, save_ohlcv_csv
from src.evaluation.backtest import run_backtest
from src.evaluation.gates import GateThresholds, evaluate_promotion
from src.evaluation.metrics import summarize_performance
from src.execution.runner import run_paper_session
from src.models.train import train_from_symbol_candles
from src.monitoring.logger import configure_logger, log_event

Track = Literal["long_term", "intraday"]


def _symbol_file(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_').upper()}_{timeframe}.csv"


def _tracks(track_arg: str) -> list[Track]:
    if track_arg == "both":
        return ["long_term", "intraday"]
    return [track_arg]  # type: ignore[list-item]


def _track_timeframe(settings: AppSettings, track: Track) -> str:
    return settings.long_term_timeframe if track == "long_term" else settings.intraday_timeframe


def _symbols_for_universe(settings: AppSettings, universe: str) -> list[str]:
    return settings.training_symbol_list if universe == "training" else settings.safe_live_symbol_list


def _synthetic_seed(symbol: str, timeframe: str) -> int:
    return sum(ord(c) for c in f"{symbol}:{timeframe}") % 9999


def _ensure_candles_for_symbol(
    settings: AppSettings,
    symbol: str,
    timeframe: str,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    data_path = settings.data_dir / timeframe / _symbol_file(symbol, timeframe)
    if not data_path.exists():
        if force_synthetic:
            candles = generate_synthetic_ohlcv(length=settings.data_limit, seed=_synthetic_seed(symbol, timeframe))
        else:
            request = FetchRequest(
                provider=settings.data_provider,
                symbol=symbol,
                timeframe=timeframe,
                limit=settings.data_limit,
                market_timezone=settings.market_timezone,
            )
            candles = fetch_ohlcv(request)
        save_ohlcv_csv(candles, data_path)
    return load_ohlcv_csv(data_path)


def _ensure_model_exists(settings: AppSettings, track: Track, force_synthetic: bool = False) -> None:
    model_path = settings.model_path_for_track(track)
    metadata_path = settings.metadata_path_for_track(track)
    if model_path.exists() and metadata_path.exists():
        return
    timeframe = _track_timeframe(settings, track)
    candles_by_symbol = {
        symbol: _ensure_candles_for_symbol(
            settings=settings,
            symbol=symbol,
            timeframe=timeframe,
            force_synthetic=force_synthetic,
        )
        for symbol in settings.training_symbol_list
    }
    train_from_symbol_candles(
        candles_by_symbol=candles_by_symbol,
        model_path=model_path,
        metadata_path=metadata_path,
        horizon=settings.prediction_horizon,
        target_return_threshold=settings.target_return_threshold,
        track_name=track,
    )


def fetch_data(args: argparse.Namespace) -> None:
    settings, _ = load_settings()
    symbols = [args.symbol.upper()] if args.symbol else _symbols_for_universe(settings, args.universe)
    for track in _tracks(args.track):
        timeframe = _track_timeframe(settings, track)
        for symbol in symbols:
            if args.synthetic:
                candles = generate_synthetic_ohlcv(length=settings.data_limit, seed=_synthetic_seed(symbol, timeframe))
            else:
                request = FetchRequest(
                    provider=settings.data_provider,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=settings.data_limit,
                    market_timezone=settings.market_timezone,
                )
                candles = fetch_ohlcv(request)
            out_path = settings.data_dir / timeframe / _symbol_file(symbol, timeframe)
            save_ohlcv_csv(candles, out_path)
            print(f"[{track}] Saved candles: {out_path} ({len(candles)} rows)")


def train_model(args: argparse.Namespace) -> None:
    settings, _ = load_settings()
    symbols = [args.symbol.upper()] if args.symbol else settings.training_symbol_list
    for track in _tracks(args.track):
        timeframe = _track_timeframe(settings, track)
        candles_by_symbol = {
            symbol: _ensure_candles_for_symbol(
                settings=settings,
                symbol=symbol,
                timeframe=timeframe,
                force_synthetic=args.synthetic,
            )
            for symbol in symbols
        }
        result = train_from_symbol_candles(
            candles_by_symbol=candles_by_symbol,
            model_path=settings.model_path_for_track(track),
            metadata_path=settings.metadata_path_for_track(track),
            horizon=settings.prediction_horizon,
            target_return_threshold=settings.target_return_threshold,
            track_name=track,
        )
        print(f"[{track}] Model trained and saved.")
        print(json.dumps(result.validation_metrics, indent=2, sort_keys=True))


def backtest(args: argparse.Namespace) -> None:
    settings, risk_limits = load_settings()
    track: Track = args.track
    timeframe = _track_timeframe(settings, track)
    symbol = (args.symbol or settings.safe_live_symbol_list[0]).upper()
    candles = _ensure_candles_for_symbol(settings, symbol, timeframe, force_synthetic=args.synthetic)
    _ensure_model_exists(settings, track=track, force_synthetic=args.synthetic)
    report = run_backtest(
        candles=candles,
        symbol=symbol,
        settings=settings,
        risk_limits=risk_limits,
        report_dir=settings.reports_dir / "backtest" / track / symbol,
        model_path=settings.model_path_for_track(track),
        metadata_path=settings.metadata_path_for_track(track),
        track_name=track,
    )
    print(f"Backtest report: {report.report_path}")
    print(f"Decision: {report.gate_decision}")
    for reason in report.gate_reasons:
        print(f"- {reason}")


def run_paper(args: argparse.Namespace) -> None:
    settings, risk_limits = load_settings()
    track: Track = args.track
    timeframe = _track_timeframe(settings, track)
    symbol = (args.symbol or settings.safe_live_symbol_list[0]).upper()
    log_file = settings.logs_dir / "paper_runner.log"
    logger = configure_logger(log_file)
    print(mode_banner(settings.mode))

    candles = _ensure_candles_for_symbol(settings, symbol, timeframe, force_synthetic=args.synthetic)
    _ensure_model_exists(settings, track=track, force_synthetic=args.synthetic)

    result = run_paper_session(
        candles=candles,
        symbol=symbol,
        settings=settings,
        risk_limits=risk_limits,
        output_dir=settings.reports_dir / "paper" / track / symbol,
        model_path=settings.model_path_for_track(track),
        metadata_path=settings.metadata_path_for_track(track),
        logger=logger,
    )
    metrics = summarize_performance(result.equity_curve, result.trade_pnls)
    gate = evaluate_promotion(metrics, GateThresholds())
    log_event(
        logger,
        "paper_run_complete",
        {
            "symbol": symbol,
            "track": track,
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
    print(f"Market: {settings.market}")
    print(f"Data provider: {settings.data_provider}")
    print(f"Execution provider: {settings.execution_provider}")
    print(f"Training symbols: {len(settings.training_symbol_list)}")
    print(f"Safe live symbols: {', '.join(settings.safe_live_symbol_list)}")
    print(f"Long-term timeframe: {settings.long_term_timeframe}")
    print(f"Intraday timeframe: {settings.intraday_timeframe}")
    print(f"Report timezone: {settings.report_timezone}")
    print(f"Risk limits: {risk_limits.model_dump()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI auto trader (paper-first) CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-data", help="Fetch market candles")
    fetch.add_argument("--symbol", type=str, default=None, help="Override symbol")
    fetch.add_argument(
        "--universe",
        choices=["training", "safe_live"],
        default="training",
        help="Symbol universe to fetch data for.",
    )
    fetch.add_argument(
        "--track",
        choices=["long_term", "intraday", "both"],
        default="both",
        help="Which timeframe tracks to fetch.",
    )
    fetch.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic candles (offline testing).",
    )
    fetch.set_defaults(func=fetch_data)

    train = sub.add_parser("train", help="Train baseline model")
    train.add_argument("--symbol", type=str, default=None, help="Override symbol")
    train.add_argument(
        "--track",
        choices=["long_term", "intraday", "both"],
        default="both",
        help="Train one or both strategy tracks.",
    )
    train.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of market data for training.",
    )
    train.set_defaults(func=train_model)

    bt = sub.add_parser("backtest", help="Run backtest + gate evaluation")
    bt.add_argument("--symbol", type=str, default=None, help="Override symbol")
    bt.add_argument(
        "--track",
        choices=["long_term", "intraday"],
        default="long_term",
        help="Select strategy track for backtesting.",
    )
    bt.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for backtest input.",
    )
    bt.set_defaults(func=backtest)

    paper = sub.add_parser("paper-run", help="Run paper trading simulation loop")
    paper.add_argument("--symbol", type=str, default=None, help="Override symbol")
    paper.add_argument(
        "--track",
        choices=["long_term", "intraday"],
        default="long_term",
        help="Select strategy track for paper run.",
    )
    paper.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for paper-run input.",
    )
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
