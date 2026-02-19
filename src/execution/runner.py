"""Paper trading runner: predict -> risk check -> execute."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import AppSettings, RiskLimits
from src.features.feature_pipeline import build_feature_frame
from src.models.predict import predict_probabilities
from src.strategy.position_sizer import size_position_notional
from src.strategy.risk_manager import RiskManager
from src.strategy.signal_engine import Signal, SignalConfig, probability_to_signal
from src.execution.paper_broker import PaperBroker


@dataclass(frozen=True)
class PaperRunResult:
    """Paper-run outputs."""

    equity_curve: pd.Series
    trade_pnls: pd.Series
    trades_path: Path
    equity_path: Path
    final_equity: float


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def run_paper_session(
    candles: pd.DataFrame,
    symbol: str,
    settings: AppSettings,
    risk_limits: RiskLimits,
    output_dir: Path,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    logger: Any | None = None,
) -> PaperRunResult:
    """Run a deterministic paper session over candle history."""
    frame, feature_columns = build_feature_frame(candles)
    valid = frame.dropna(subset=feature_columns).reset_index(drop=True)
    valid["up_probability"] = predict_probabilities(
        valid,
        model_path=model_path or settings.model_path,
        metadata_path=metadata_path or settings.metadata_path,
    )

    broker = PaperBroker(
        initial_cash=settings.initial_cash,
        fee_rate=settings.fee_rate,
        slippage_rate=settings.slippage_rate,
    )
    risk_manager = RiskManager(risk_limits)
    signal_cfg = SignalConfig(
        long_threshold=settings.model_probability_threshold,
        short_threshold=1 - settings.model_probability_threshold,
        allow_short=False,
    )

    equity_rows: list[dict[str, Any]] = []
    for row in valid.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp).to_pydatetime()
        price = float(row.close)
        atr_value = max(float(row.atr_14), 1e-9)
        equity_before = broker.portfolio_value({symbol: price})
        signal = probability_to_signal(float(row.up_probability), signal_cfg)

        symbol_exposure = broker.symbol_exposure(symbol, price)
        total_exposure = broker.total_exposure({symbol: price})

        if signal == Signal.LONG and symbol_exposure <= 1e-9:
            order_notional = size_position_notional(
                equity=equity_before,
                atr_value=atr_value,
                price=price,
                max_position_pct=risk_limits.max_position_pct,
                stop_loss_atr_multiplier=risk_limits.stop_loss_atr_multiplier,
            )
            ok, reason = risk_manager.can_place_order(
                side="buy",
                order_notional=order_notional,
                equity=equity_before,
                symbol_exposure=symbol_exposure,
                total_exposure=total_exposure,
                open_positions=broker.open_positions(),
                timestamp=ts,
            )
            if ok:
                broker.place_order(symbol=symbol, side="buy", price=price, notional=order_notional, timestamp=ts)
            elif logger is not None:
                logger.info(f'{{"event":"risk_block","timestamp":"{ts.isoformat()}","reason":"{reason}"}}')

        if signal == Signal.FLAT and symbol_exposure > 1e-9:
            broker.place_order(symbol=symbol, side="sell", price=price, notional=symbol_exposure, timestamp=ts)

        equity_after = broker.portfolio_value({symbol: price})
        risk_manager.update_equity(equity_after, ts)
        ts_utc = _to_utc_timestamp(ts)
        equity_rows.append(
            {
                "timestamp_utc": ts_utc.to_pydatetime(),
                "timestamp_report_tz": ts_utc.tz_convert(settings.report_timezone).to_pydatetime(),
                "price": price,
                "signal": signal.value,
                "up_probability": float(row.up_probability),
                "equity": equity_after,
            },
        )

    if not valid.empty:
        last_row = valid.iloc[-1]
        last_ts = pd.Timestamp(last_row["timestamp"]).to_pydatetime()
        last_price = float(last_row["close"])
        last_exposure = broker.symbol_exposure(symbol, last_price)
        if last_exposure > 1e-9:
            broker.place_order(
                symbol=symbol,
                side="sell",
                price=last_price,
                notional=last_exposure,
                timestamp=last_ts,
            )
            last_ts_utc = _to_utc_timestamp(last_ts)
            equity_rows.append(
                {
                    "timestamp_utc": last_ts_utc.to_pydatetime(),
                    "timestamp_report_tz": last_ts_utc.tz_convert(settings.report_timezone).to_pydatetime(),
                    "price": last_price,
                    "signal": "forced_flatten",
                    "up_probability": float(last_row["up_probability"]),
                    "equity": broker.portfolio_value({symbol: last_price}),
                },
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "paper_trades.csv"
    equity_path = output_dir / "paper_equity_curve.csv"
    broker.export_trade_journal(trades_path)
    equity_df = pd.DataFrame(equity_rows)
    equity_df.to_csv(equity_path, index=False)

    equity_curve = equity_df["equity"] if not equity_df.empty else pd.Series(dtype=float)
    trade_pnls = pd.Series(broker.realized_pnls, dtype=float)
    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else settings.initial_cash
    return PaperRunResult(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        trades_path=trades_path,
        equity_path=equity_path,
        final_equity=final_equity,
    )
