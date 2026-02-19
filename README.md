# Auto Trader (Paper-First AI Bot Foundation)

This repository provides a **paper-trading-first AI auto trading foundation** inspired by battle-tested open-source trading frameworks (especially the workflow style used in Freqtrade/FreqAI ecosystems).

> Safety policy: default mode is **paper** and live mode is blocked unless strict acknowledgements are explicitly set.

---

## What this project includes

- Data ingestion and synthetic data fallback.
- Leakage-aware feature engineering on OHLCV candles.
- Baseline ML training (logistic regression classifier).
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

Edit `.env` as needed (symbols, timeframe, data size, thresholds).

## 3) Validate mode and risk config

```bash
python3 -m src.cli status
```

Expected: `PAPER TRADING MODE (safe default)`

## 4) Fetch market data (or generate synthetic test data)

```bash
python3 -m src.cli fetch-data
# or offline:
python3 -m src.cli fetch-data --synthetic
```

## 5) Train baseline model

```bash
python3 -m src.cli train
```

Outputs:
- `models/model.joblib`
- `models/model_metadata.json`

## 6) Run backtest and promotion gate evaluation

```bash
python3 -m src.cli backtest
```

Outputs:
- `reports/backtest/backtest_report.json`
- `reports/backtest/paper_equity_curve.csv`
- `reports/backtest/paper_trades.csv`

## 7) Run paper trading simulation

```bash
python3 -m src.cli paper-run
```

Outputs:
- `reports/paper/paper_equity_curve.csv`
- `reports/paper/paper_trades.csv`
- `logs/paper_runner.log`

---

## Makefile shortcuts

```bash
make install-dev
make test
make fetch-data
make train
make backtest
make paper-run
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
- real exchange execution hardening
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

---

## Continuous improvement playbook

1. Use walk-forward retraining windows.
2. Validate across different market regimes (trend/chop/volatile).
3. Track feature drift over time.
4. Scale position size by confidence and volatility.
5. Run A/B paper portfolios for challenger strategies.
6. Review every loss by cause (signal, execution, or risk handling).
7. Do tiny-capital staged rollout only after sustained paper consistency.

---

## Running tests

```bash
python3 -m pytest
```

---

## Important disclaimer

This software is for educational and research use.  
Trading carries substantial financial risk. Past results do not guarantee future outcomes.