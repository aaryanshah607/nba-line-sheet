# The Line Sheet

A data terminal for NBA game history — records, moneylines, and per-team splits in one dashboard. Modeled on the Yahoo Finance / brokerage-terminal UX pattern: aggregation and history, not predictions or picks.

**No betting recommendations, no expected-value claims, no model output.** This project is a data engineering and front-end build, not a prediction system.

![screenshot placeholder](docs/screenshot.png)

## What it does

- Ingests and cleans historical NBA game data: moneylines, spreads, totals, scores, and dates (2007–2026, 23,586 games)
- Builds point-in-time team features (rolling win %, rest days) with strict no-lookahead discipline
- Serves a React dashboard: sortable standings, a live moneyline ticker, and click-through team pages showing recent game-by-game line history

## Why

Most "AI sports betting" projects lead with a prediction claim that's hard to back up honestly. This one doesn't make one. The goal is a clean, well-engineered data product — the sports-data equivalent of a finance terminal — that surfaces history and context clearly, and leaves any judgment calls to the person reading it.

## Project structure

```
nba-line-sheet/
├── data/
│   ├── nba_odds_clean.csv       # cleaned game-level odds, 2007-2026
│   ├── nba_features.csv         # + point-in-time rolling features
│   └── dashboard_compact.json   # compact array format used by the dashboard
├── src/
│   ├── build_features.py        # leakage-free feature engineering
│   ├── train_baseline.py        # baseline model vs. market (see Appendix)
│   └── backtest.py              # EV/Kelly backtest (see Appendix)
├── dashboard/
│   └── LineSheet.jsx            # React dashboard component
└── README.md
```

## Data source

Historical odds, scores, and moneylines aggregated from public game-log and sportsbook-odds data (2007–08 season through the current 2025–26 season, 23,586 total games after deduplication). Every row includes home/away teams, date, moneyline for both sides, spread, total, final score, and days of rest.

## Running the dashboard

The dashboard is a self-contained React component (`dashboard/LineSheet.jsx`) with the season's data embedded directly — no backend or database required. Drop it into any React environment (Create React App, Vite, Next.js) that supports JSX, or open it directly in a React-capable sandbox.

## Running the data pipeline

```bash
pip install -r requirements.txt
python src/build_features.py   # rebuilds nba_features.csv from nba_odds_clean.csv
```

`build_features.py` computes rolling win percentages and rest days for each team using only games that occurred strictly before the game being featured — this avoids the lookahead leakage that silently inflates a lot of sports-model backtests.

## Appendix: prediction backtest (research notes, not part of the product)

Before landing on the data-terminal direction, I tested whether a simple baseline model (logistic regression on rolling win % and rest-day features) could out-predict the market's de-vigged implied probability. `train_baseline.py` and `backtest.py` contain that experiment.

**Honest result:** on a strict chronological out-of-sample split, the model did not beat the market (Brier score 0.220 vs. market's 0.198; AUC 0.700 vs. 0.758), and a Kelly-sized betting simulation using the model's edge lost money out of sample. This is the expected outcome for a small feature set against an efficient market, and it's the reason the project pivoted away from a "prediction" framing toward pure data presentation.

These scripts are included for transparency about the full research process, not as a claim of a working betting strategy.

## Disclaimer

This project is for data engineering and educational purposes. It does not provide betting advice, predictions, or recommendations. No part of this repository facilitates or encourages wagering.
