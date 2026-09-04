"""
Build point-in-time features for each game from the game log itself.
Critical rule: every feature for a game must only use information
available strictly BEFORE that game's date. No lookahead.
"""
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, rolling_windows=(5, 10, 20)) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    # Long format: one row per (team, game) so we can compute rolling
    # team-level stats chronologically without leaking future games.
    home = df[["Date", "Home", "Away", "home_win"]].copy()
    home["team"] = home["Home"]
    home["opponent"] = home["Away"]
    home["win"] = home["home_win"]
    home["is_home"] = 1

    away = df[["Date", "Home", "Away", "home_win"]].copy()
    away["team"] = away["Away"]
    away["opponent"] = away["Home"]
    away["win"] = 1 - away["home_win"]
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"]).reset_index(drop=True)

    # Rolling win pct BEFORE this game (shift(1) avoids leakage of the
    # game's own outcome into its own feature)
    for w in rolling_windows:
        col = f"win_pct_last{w}"
        long_df[col] = (
            long_df.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    # Season-to-date win pct before this game
    long_df["season_win_pct"] = (
        long_df.groupby("team")["win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # Days since team's last game (rest)
    long_df["prev_date"] = long_df.groupby("team")["Date"].shift(1)
    long_df["days_rest"] = (long_df["Date"] - long_df["prev_date"]).dt.days
    long_df["days_rest"] = long_df["days_rest"].fillna(7).clip(upper=10)

    feat_cols = [f"win_pct_last{w}" for w in rolling_windows] + [
        "season_win_pct", "days_rest"
    ]

    home_feats = long_df[long_df["is_home"] == 1][
        ["Date", "team", "opponent"] + feat_cols
    ].rename(columns={c: f"home_{c}" for c in feat_cols})
    home_feats = home_feats.rename(columns={"team": "Home", "opponent": "Away"})

    away_feats = long_df[long_df["is_home"] == 0][
        ["Date", "team", "opponent"] + feat_cols
    ].rename(columns={c: f"away_{c}" for c in feat_cols})
    away_feats = away_feats.rename(columns={"team": "Away", "opponent": "Home"})

    merged = df.merge(home_feats, on=["Date", "Home", "Away"], how="left")
    merged = merged.merge(away_feats, on=["Date", "Home", "Away"], how="left")

    # diff features (home minus away) - the actual model inputs
    for w in rolling_windows:
        merged[f"diff_win_pct_last{w}"] = (
            merged[f"home_win_pct_last{w}"] - merged[f"away_win_pct_last{w}"]
        )
    merged["diff_season_win_pct"] = (
        merged["home_season_win_pct"] - merged["away_season_win_pct"]
    )
    merged["diff_days_rest"] = merged["home_days_rest"] - merged["away_days_rest"]

    return merged


if __name__ == "__main__":
    df = pd.read_csv("data/nba_odds_clean.csv", parse_dates=["Date"])
    out = build_features(df)
    # drop first-few-games-of-history rows where rolling stats are NaN
    model_cols = [c for c in out.columns if c.startswith("diff_")]
    out = out.dropna(subset=model_cols).reset_index(drop=True)
    out.to_csv("data/nba_features.csv", index=False)
    print(f"Built features for {len(out)} games with no lookahead leakage.")
    print(out[["Date", "Home", "Away"] + model_cols + ["market_home_prob", "home_win"]].tail(5))
