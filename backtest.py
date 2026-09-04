"""
Step 4/7: EV engine + simple backtest.
Rule: only bet when model probability exceeds market (de-vigged) probability
by more than `edge_threshold`, sized with a fractional-Kelly stake.
This directly tests the "Do we have a genuine, repeatable edge?" question.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from train_baseline import FEATURES, season_of

def american_payout(ml):
    ml = np.asarray(ml, dtype=float)
    # profit per $1 staked if the bet wins
    return np.where(ml > 0, ml / 100, 100 / -ml)

def kelly_fraction(p, b, fraction=0.25):
    # b = net odds (profit per $1 staked); f* = (p*b - (1-p)) / b
    f_star = (p * b - (1 - p)) / b
    return np.clip(f_star, 0, None) * fraction  # fractional Kelly, cap losses off at 0

def run_backtest(edge_threshold=0.04, kelly_mult=0.25, starting_bankroll=10_000):
    df = pd.read_csv("data/nba_features.csv", parse_dates=["Date"])
    df["season"] = df["Date"].apply(season_of)

    train = df[df["season"] <= 2022]
    test = df[df["season"] >= 2023].copy().sort_values("Date").reset_index(drop=True)

    model = LogisticRegression(max_iter=1000)
    model.fit(train[FEATURES], train["home_win"])
    test["model_prob_home"] = model.predict_proba(test[FEATURES])[:, 1]

    bankroll = starting_bankroll
    bankroll_curve = [bankroll]
    bets = []

    for _, row in test.iterrows():
        # consider both sides of the game
        for side, model_p, ml, opp_ml in [
            ("home", row["model_prob_home"], row["ML_Home"], row["ML_Away"]),
            ("away", 1 - row["model_prob_home"], row["ML_Away"], row["ML_Home"]),
        ]:
            market_p_side = row["market_home_prob"] if side == "home" else 1 - row["market_home_prob"]
            edge = model_p - market_p_side
            if edge > edge_threshold:
                b = american_payout(ml)
                stake_frac = kelly_fraction(model_p, b, kelly_mult)
                stake = bankroll * stake_frac
                if stake <= 0:
                    continue
                won = (row["home_win"] == 1) if side == "home" else (row["home_win"] == 0)
                pnl = stake * b if won else -stake
                bankroll += pnl
                bets.append({
                    "Date": row["Date"], "side": side, "edge": edge,
                    "stake": stake, "won": won, "pnl": pnl, "bankroll": bankroll
                })
        bankroll_curve.append(bankroll)

    bets_df = pd.DataFrame(bets)
    if len(bets_df) == 0:
        print("No bets triggered at this edge threshold.")
        return

    total_staked = bets_df["stake"].sum()
    total_pnl = bets_df["pnl"].sum()
    roi = total_pnl / total_staked
    win_rate = bets_df["won"].mean()
    peak = np.maximum.accumulate(bankroll_curve)
    drawdown = (np.array(bankroll_curve) - peak) / peak
    max_dd = drawdown.min()

    print(f"Edge threshold: {edge_threshold}, Kelly multiplier: {kelly_mult}")
    print(f"Bets placed: {len(bets_df)}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Total staked: ${total_staked:,.0f}")
    print(f"Total PnL: ${total_pnl:,.0f}")
    print(f"ROI (pnl/staked): {roi:.2%}")
    print(f"Final bankroll: ${bankroll:,.0f} (started at ${starting_bankroll:,.0f})")
    print(f"Max drawdown: {max_dd:.2%}")

    return bets_df, bankroll_curve

if __name__ == "__main__":
    run_backtest()
