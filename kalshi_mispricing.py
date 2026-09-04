"""
Kalshi vs. sportsbook divergence — a research/labeling tool, NOT a betting signal.

What this does:
  For today's NBA games, compare Kalshi's prediction-market price (the
  crowd's implied probability of a home win) against the sportsbook
  consensus de-vigged implied probability already computed in
  data/nba_odds_clean.csv (`market_home_prob`).

  Where the two disagree by more than a threshold, that's flagged as a
  "divergence" — nothing more. This does NOT claim either market is
  wrong, does NOT use our own baseline model (which underperformed the
  market in backtest.py and should not be treated as a source of edge),
  and does NOT compute expected value or suggest a position.

Why this is a reasonable thing to show:
  Kalshi is a comparatively thin, newer market. Sportsbooks are a much
  older, deeper, more heavily arbitraged one. When the two disagree
  meaningfully, that's an interesting, honest fact to surface on a data
  terminal — the same way a finance terminal shows when two exchanges'
  quotes for the same asset briefly diverge. It's context, not advice.

API reference:
  Kalshi public market-data endpoints do not require authentication.
  Base URL: https://api.elections.kalshi.com/trade-api/v2
  NBA markets live under the series ticker "KXNBA" (single-game winner
  markets); verify against Kalshi's current series listing before
  relying on this, since exchanges add/rename series over time:
      GET /series/KXNBA
      GET /markets?series_ticker=KXNBA&status=open
  Kalshi prices are integers on a 0-100 scale representing cents /
  implied probability (e.g. a "yes" price of 62 means the market is
  pricing that outcome at 62%). Do not treat these as American odds.

Usage:
  python src/kalshi_mispricing.py --demo
      Runs against a small bundled sample so you can see the output
      shape without any network access or API key.

  python src/kalshi_mispricing.py
      Hits the live Kalshi public endpoints for today's NBA markets and
      compares against the most recent sportsbook line for each
      matched team pair in data/nba_odds_clean.csv. Requires network
      access to api.elections.kalshi.com (no API key needed for this
      read-only market data).
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import requests

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
NBA_SERIES_TICKER = "KXNBAGAME"  # single-game winner markets (NOT "KXNBA",
# which is the season-championship futures market - different product)
DIVERGENCE_THRESHOLD = 0.07  # flag when markets disagree by 7+ percentage points

# Kalshi's NBA game markets label each team by city/market name only (e.g.
# "yes_sub_title": "Oklahoma City"), not by mascot. Map those labels to the
# full team names used in our own odds data. Confirmed against a live
# Kalshi response on 2026-09-03; the two LA franchises are the one spot
# worth double-checking again if this ever looks wrong, since Kalshi may
# label the Lakers as "Los Angeles" and the Clippers as "LA Clippers" or
# vice versa - verify against a live response before trusting blindly.
KALSHI_CITY_TO_TEAM = {
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "Golden State": "Golden State Warriors",
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "Los Angeles": "Los Angeles Lakers",
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
}


@dataclass
class Divergence:
    matchup: str
    kalshi_prob: float
    sportsbook_prob: float
    divergence: float
    kalshi_ticker: str

    def as_dict(self):
        return {
            "matchup": self.matchup,
            "kalshi_implied_prob": round(self.kalshi_prob, 3),
            "sportsbook_implied_prob": round(self.sportsbook_prob, 3),
            "divergence_pct_pts": round(self.divergence * 100, 1),
            "kalshi_ticker": self.kalshi_ticker,
        }


def fetch_open_nba_markets(session: requests.Session) -> list[dict]:
    """Pull today's open NBA single-game markets from Kalshi's public API."""
    resp = session.get(
        f"{KALSHI_BASE_URL}/markets",
        params={"series_ticker": NBA_SERIES_TICKER, "status": "open"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("markets", [])


def group_markets_by_event(markets: list[dict]) -> dict[str, list[dict]]:
    """Each NBA game is two markets (one per team) sharing an event_ticker."""
    events = defaultdict(list)
    for m in markets:
        events[m["event_ticker"]].append(m)
    return events


def kalshi_implied_prob(market: dict) -> float | None:
    """Midpoint of yes bid/ask, as a probability. Kalshi returns these as
    decimal-dollar strings (e.g. "0.5500"), not the 0-100 cent scale some
    docs/examples show - confirmed against a live response."""
    bid = market.get("yes_bid_dollars")
    ask = market.get("yes_ask_dollars")
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def compute_divergences(
    markets: list[dict], odds_df: pd.DataFrame
) -> list[Divergence]:
    results = []
    events = group_markets_by_event(markets)

    for event_ticker, event_markets in events.items():
        if len(event_markets) != 2:
            continue  # skip anything that isn't a clean two-sided game market

        sides = {}
        for m in event_markets:
            city = m.get("yes_sub_title", "")
            team = KALSHI_CITY_TO_TEAM.get(city)
            if team is None:
                continue  # unmapped label - skip rather than guess
            prob = kalshi_implied_prob(m)
            if prob is None:
                continue
            sides[team] = (prob, m.get("ticker", ""))

        if len(sides) != 2:
            continue

        (team_a, (prob_a, ticker_a)), (team_b, (prob_b, ticker_b)) = sides.items()

        # find this exact matchup in our odds data (either home/away order)
        # to learn which side is "home" for the sportsbook probability
        pair_matches = odds_df[
            ((odds_df["Home"] == team_a) & (odds_df["Away"] == team_b))
            | ((odds_df["Home"] == team_b) & (odds_df["Away"] == team_a))
        ]
        if pair_matches.empty:
            continue
        row = pair_matches.sort_values("Date").iloc[-1]
        home, away = row["Home"], row["Away"]
        sportsbook_home_prob = row["market_home_prob"]

        kalshi_home_prob = prob_a if team_a == home else prob_b
        home_ticker = ticker_a if team_a == home else ticker_b

        divergence = abs(kalshi_home_prob - sportsbook_home_prob)
        if divergence >= DIVERGENCE_THRESHOLD:
            results.append(
                Divergence(
                    matchup=f"{away} @ {home}",
                    kalshi_prob=kalshi_home_prob,
                    sportsbook_prob=sportsbook_home_prob,
                    divergence=divergence,
                    kalshi_ticker=home_ticker,
                )
            )

    results.sort(key=lambda d: d.divergence, reverse=True)
    return results


def demo_run(odds_df: pd.DataFrame) -> list[Divergence]:
    """Two-sided sample games standing in for a live Kalshi response, so
    the script is runnable and demoable without network access."""
    sample_markets = [
        {"event_ticker": "DEMO-LALBOS", "yes_sub_title": "Los Angeles",
         "yes_bid_dollars": "0.4600", "yes_ask_dollars": "0.4800", "ticker": "DEMO-LAL"},
        {"event_ticker": "DEMO-LALBOS", "yes_sub_title": "Boston",
         "yes_bid_dollars": "0.5100", "yes_ask_dollars": "0.5300", "ticker": "DEMO-BOS"},
        {"event_ticker": "DEMO-DETCHI", "yes_sub_title": "Detroit",
         "yes_bid_dollars": "0.7900", "yes_ask_dollars": "0.8100", "ticker": "DEMO-DET"},
        {"event_ticker": "DEMO-DETCHI", "yes_sub_title": "Chicago",
         "yes_bid_dollars": "0.1800", "yes_ask_dollars": "0.2000", "ticker": "DEMO-CHI"},
    ]
    return compute_divergences(sample_markets, odds_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run against bundled sample data, no network needed")
    parser.add_argument("--odds-csv", default="data/nba_odds_clean.csv")
    parser.add_argument("--out", default=None, help="optional path to write results JSON")
    args = parser.parse_args()

    odds_df = pd.read_csv(args.odds_csv, parse_dates=["Date"])

    if args.demo:
        divergences = demo_run(odds_df)
    else:
        session = requests.Session()
        try:
            markets = fetch_open_nba_markets(session)
        except requests.RequestException as e:
            print(f"Could not reach Kalshi API: {e}", file=sys.stderr)
            print("Try --demo to see expected output shape without network access.", file=sys.stderr)
            sys.exit(1)
        divergences = compute_divergences(markets, odds_df)

    if not divergences:
        print("No divergences at or above threshold found.")
        return

    print(f"{'Matchup':<40} {'Kalshi':>8} {'Sportsbook':>12} {'Diverg.':>9}")
    for d in divergences:
        print(
            f"{d.matchup:<40} {d.kalshi_prob:>7.1%} {d.sportsbook_prob:>11.1%} "
            f"{d.divergence * 100:>8.1f}pp"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump([d.as_dict() for d in divergences], f, indent=2)
        print(f"\nWrote {len(divergences)} divergences to {args.out}")


if __name__ == "__main__":
    main()
