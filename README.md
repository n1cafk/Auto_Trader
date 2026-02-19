# Auto Trader (Paper-First US ETFs + Stocks Bot Foundation)

This repository provides a **paper-trading-first AI trading bot foundation** focused on a safer beginner path:

- **US ETFs + large-cap stocks** for training
- **safe live rollout universe** (ETF-only by default)
- **long-term + intraday model tracks**
- strict risk gates before any real-capital deployment

> Safety policy: default mode is **paper**, with live mode blocked unless strict acknowledgements are explicitly set.

---

## What this project includes

- Data ingestion for US equities (default provider: `yfinance`) + synthetic fallback.
- Leakage-aware feature engineering on OHLCV candles.
- Multi-symbol baseline ML training (logistic regression classifier).
- Two strategy tracks:
  - `long_term` (default `1d`)
  - `intraday` (default `15m`)
- Model artifact registry (model + metadata).
- Signal engine (probability → long/flat/short policy).
- Risk manager guardrails:
  - per-position caps
  - max portfolio exposure
  - max daily loss cutoff
  - max drawdown emergency stop
- Paper broker execution simulator (fees/slippage, positions, trade journal).
- Backtest metrics + promotion gates:
  - Sharpe / Sortino
  - drawdown
  - trade count / win rate
- Structured logging and alert hook stubs.

---

## Project structure

```text
src/
  config/       settings + risk limits
  dashboard/    Streamlit monitoring frontend
  data/         market data and dataset building
  features/     indicators and feature pipeline
  models/       train / predict / artifact registry
  strategy/     signal, sizing, and risk logic
  execution/    paper broker and runner loop
  evaluation/   metrics, gates, backtest reports
  monitoring/   structured logs and alert stubs
tests/
  unit/
  integration/
```

---

## Quick start

## 1) Install

```bash
python3 -m pip install -e ".[dev]"
```

## 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed (training universe, safe live universe, timeframes, thresholds).

## 3) Validate mode and risk config

```bash
python3 -m src.cli status
```

Expected: `PAPER TRADING MODE (safe default)`

## 4) Fetch market data (or generate synthetic test data)

```bash
python3 -m src.cli fetch-data --universe training --track both
# or offline:
python3 -m src.cli fetch-data --synthetic --universe training --track both
```

## 5) Train baseline models (both tracks)

```bash
python3 -m src.cli train --track both
```

Outputs:
- `models/long_term_model.joblib`
- `models/long_term_model_metadata.json`
- `models/intraday_model.joblib`
- `models/intraday_model_metadata.json`

## 6) Run backtest and promotion gate evaluation

```bash
python3 -m src.cli backtest --track long_term --symbol SPY
```

Outputs:
- `reports/backtest/<track>/<symbol>/backtest_report.json`
- `reports/backtest/<track>/<symbol>/paper_equity_curve.csv`
- `reports/backtest/<track>/<symbol>/paper_trades.csv`

## 7) Run paper trading simulation

```bash
python3 -m src.cli paper-run --track intraday --symbol SPY
```

Outputs:
- `reports/paper/<track>/<symbol>/paper_equity_curve.csv`
- `reports/paper/<track>/<symbol>/paper_trades.csv`
- `logs/paper_runner.log`

## 8) Open the quick monitoring dashboard

```bash
make dashboard
# or:
streamlit run src/dashboard/app.py
```

Dashboard highlights:
- runtime status (mode/market/provider/risk limits)
- latest backtest gate decision
- model metadata by track
- paper equity curve chart + trade journal preview
- recent runner logs

---

## Makefile shortcuts

```bash
make install-dev
make test
make fetch-data
make train
make backtest
make paper-run
make dashboard
```

---

## Safety controls for live mode

Live mode is blocked by default and requires both:

```env
MODE=live
LIVE_TRADING_ENABLED=true
LIVE_ACKNOWLEDGEMENT=I_UNDERSTAND_LIVE_TRADING_RISK
```

Even after these flags, this project should still be considered a **research/paper-trading system** until extended with:
- real broker execution hardening
- incident monitoring
- kill-switch orchestration
- production reliability engineering

---

## Objective promotion gates (paper → candidate live trial)

The default gate evaluator checks:
- minimum Sharpe and Sortino
- maximum drawdown
- minimum trade count
- minimum win rate

If gates fail, decision is one of:
- `retrain_required`
- `paper_continue`
- `candidate_live_trial`

## Recommended beginner rollout

1. Train on broad universe (ETFs + large caps).
2. Paper trade at least 45 days.
3. Start real capital only on safe universe (`SPY,VTI,QQQ` by default).
4. Keep long-only, no leverage, strict risk limits.

---

## Continuous improvement playbook

1. Use walk-forward retraining windows.
2. Validate across different market regimes (trend/chop/volatile).
3. Track feature drift over time.
4. Scale position size by confidence and volatility.
5. Run A/B paper portfolios for challenger strategies.
6. Review every loss by cause (signal, execution, or risk handling).
7. Do tiny-capital staged rollout only after sustained paper consistency.
8. Promote intraday track to live only after long-term track shows stable behavior.

---

## Running tests

```bash
python3 -m pytest
```

---

## Important disclaimer

This software is for educational and research use.  
Trading carries substantial financial risk. Past results do not guarantee future outcomes.