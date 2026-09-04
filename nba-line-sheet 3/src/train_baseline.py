"""
Step 3/4/6 of the roadmap: baseline model, EV engine, and backtest.
Question we're answering: can a simple, interpretable model produce
better-calibrated win probabilities than the market's de-vigged price?

Strict chronological split (by season) - no shuffling, no leakage:
  train:      2007-08 .. 2023-24
  validation: 2024-25
  test/paper: 2025-26 (partial season, most recent games)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

FEATURES = [
    "diff_win_pct_last5",
    "diff_win_pct_last10",
    "diff_win_pct_last20",
    "diff_season_win_pct",
    "diff_days_rest",
]

def season_of(date):
    # NBA season starts ~October; games Jan-Jun belong to the season
    # that started the previous October
    if date.month >= 10:
        return date.year
    return date.year - 1

def main():
    df = pd.read_csv("data/nba_features.csv", parse_dates=["Date"])
    df["season"] = df["Date"].apply(season_of)

    train = df[df["season"] <= 2022].copy()          # 2007-08 .. 2022-23
    val = df[df["season"] == 2023].copy()             # 2023-24
    test = df[df["season"] >= 2024].copy()             # 2024-25 + 2025-26 so far

    print(f"train: {len(train)} games ({train['Date'].min().date()} - {train['Date'].max().date()})")
    print(f"val:   {len(val)} games ({val['Date'].min().date()} - {val['Date'].max().date()})")
    print(f"test:  {len(test)} games ({test['Date'].min().date()} - {test['Date'].max().date()})")

    X_train, y_train = train[FEATURES], train["home_win"]
    X_val, y_val = val[FEATURES], val["home_win"]
    X_test, y_test = test[FEATURES], test["home_win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    for name, X, y, split in [("VALIDATION", X_val, y_val, val), ("TEST/PAPER", X_test, y_test, test)]:
        model_prob = model.predict_proba(X)[:, 1]
        market_prob = split["market_home_prob"].values

        print(f"\n=== {name} ({len(X)} games) ===")
        print(f"Model  - LogLoss: {log_loss(y, model_prob):.4f}  Brier: {brier_score_loss(y, model_prob):.4f}  AUC: {roc_auc_score(y, model_prob):.4f}")
        print(f"Market - LogLoss: {log_loss(y, market_prob):.4f}  Brier: {brier_score_loss(y, market_prob):.4f}  AUC: {roc_auc_score(y, market_prob):.4f}")

        edge = model_prob - market_prob
        print(f"Mean |model - market| prob diff: {np.abs(edge).mean():.4f}")

    print("\nModel coefficients (feature -> weight):")
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:28s} {c:+.4f}")

    return model, df

if __name__ == "__main__":
    main()
