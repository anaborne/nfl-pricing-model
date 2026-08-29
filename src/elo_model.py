"""Walk-forward NFL Elo, pricing every 2025 regular-season game with zero lookahead: a
game's probability uses only results and lines available before it was played.

The rating update, MOV dampener, HFA constant and season-carryover rule follow the
publicly documented 538-style NFL Elo family. K=20, HFA=48 and 1/3 regression are that
family's standard published values and were not tuned on 2025. Tuning constants on the
season you then grade against is the easiest way to make a backtest dishonest, so the
constants were fixed once and whatever came out was reported.

Data is nflverse/nfldata games.csv (https://github.com/nflverse/nfldata). nflverse
documents its odds fields only as "Odds for away/home team to win the game" and does not
say whether they are opening or closing quotes, so this repo calls them the recorded line
and never "the close".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

K_FACTOR = 20.0
HOME_FIELD_ADVANTAGE = 48.0  # Elo points, added to the home side before the win prob
SEASON_REGRESSION = 1.0 / 3.0  # fraction of the gap to 1500 removed between seasons
BASE_RATING = 1500.0
WARMUP_FIRST_SEASON = 2002  # first season of the current 32-team, post-realignment NFL

# Relocated franchises keep one continuous rating under their current abbreviation.
FRANCHISE_ALIASES = {
    "OAK": "LV",   # Raiders: Oakland -> Las Vegas, 2020
    "SD": "LAC",   # Chargers: San Diego -> Los Angeles, 2017
    "STL": "LA",   # Rams: St. Louis -> Los Angeles, 2016
}


def _canon(team: str) -> str:
    return FRANCHISE_ALIASES.get(team, team)


def elo_win_prob(elo_diff: float) -> float:
    """Standard logistic Elo win-probability formula (400-point scale)."""
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def mov_multiplier(margin: int, elo_diff_winner_minus_loser: float) -> float:
    """Blowouts move ratings more than squeakers, damped by the elo gap: running up the
    score on a team you were already heavy against says less than doing it as a dog.
    """
    return math.log(abs(margin) + 1) * (2.2 / (0.001 * elo_diff_winner_minus_loser + 2.2))


@dataclass(frozen=True)
class EloParams:
    """Bundled so src/sensitivity.py can replay the same walk-forward under different
    values without copying the loop. The defaults are the published configuration.
    """

    k_factor: float = K_FACTOR
    home_field_advantage: float = HOME_FIELD_ADVANTAGE
    season_regression: float = SEASON_REGRESSION


@dataclass
class EloState:
    params: EloParams = field(default_factory=EloParams)
    ratings: dict[str, float] = field(default_factory=dict)
    seasons_seen: dict[str, int] = field(default_factory=dict)

    def get(self, team: str, season: int) -> float:
        team = _canon(team)
        if team not in self.ratings:
            self.ratings[team] = BASE_RATING
            self.seasons_seen[team] = season
            return BASE_RATING
        last_season = self.seasons_seen[team]
        if season > last_season:
            gap = season - last_season
            rating = self.ratings[team]
            for _ in range(gap):
                rating = rating + self.params.season_regression * (BASE_RATING - rating)
            self.ratings[team] = rating
            self.seasons_seen[team] = season
        return self.ratings[team]

    def update(self, home: str, away: str, season: int, home_margin: int) -> None:
        home, away = _canon(home), _canon(away)
        home_elo = self.get(home, season)
        away_elo = self.get(away, season)
        elo_diff = (home_elo + self.params.home_field_advantage) - away_elo
        prob_home_win = elo_win_prob(elo_diff)
        actual_home = 1.0 if home_margin > 0 else (0.5 if home_margin == 0 else 0.0)

        winner_elo = home_elo if home_margin >= 0 else away_elo
        loser_elo = away_elo if home_margin >= 0 else home_elo
        mult = mov_multiplier(home_margin if home_margin != 0 else 1, winner_elo - loser_elo)

        delta = self.params.k_factor * mult * (actual_home - prob_home_win)
        self.ratings[home] = home_elo + delta
        self.ratings[away] = away_elo - delta


def load_games(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["game_type"] == "REG"].copy()
    df = df[df["season"] >= WARMUP_FIRST_SEASON].copy()
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    return df


def run_walkforward(
    games: pd.DataFrame, price_season: int, params: EloParams | None = None
) -> pd.DataFrame:
    """Replay every game in order, updating Elo after each. Games in `price_season` are
    recorded at their pre-game probability, before their own result is folded in. Earlier
    seasons are warmup: they move ratings but are never priced or graded.
    """
    state = EloState(params=params or EloParams())
    rows = []
    for _, g in games.iterrows():
        season = int(g["season"])
        home, away = g["home_team"], g["away_team"]
        home_score, away_score = g["home_score"], g["away_score"]
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        home_margin = int(home_score) - int(away_score)

        if season == price_season:
            home_elo = state.get(home, season)
            away_elo = state.get(away, season)
            elo_diff = (home_elo + state.params.home_field_advantage) - away_elo
            model_prob_home = elo_win_prob(elo_diff)
            rows.append(
                {
                    "game_id": g["game_id"],
                    "season": season,
                    "week": g["week"],
                    "gameday": g["gameday"],
                    "away_team": away,
                    "home_team": home,
                    "away_score": int(away_score),
                    "home_score": int(home_score),
                    "home_win": 1 if home_margin > 0 else 0,
                    "tie": 1 if home_margin == 0 else 0,
                    "home_elo_pre": home_elo,
                    "away_elo_pre": away_elo,
                    "model_prob_home": model_prob_home,
                    "home_moneyline": g["home_moneyline"],
                    "away_moneyline": g["away_moneyline"],
                    "spread_line": g["spread_line"],
                }
            )

        state.update(home, away, season, home_margin)

    return pd.DataFrame(rows)


def moneyline_to_implied_prob(ml: float) -> float:
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Scale both raw implied probabilities so they sum to 1. Every graded number in this
    repo uses this method; src/sensitivity.py runs the two alternatives below against it.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def devig_additive(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Additive de-vig: split the overround evenly and subtract half from each side."""
    half_overround = (prob_a + prob_b - 1.0) / 2.0
    return prob_a - half_overround, prob_b - half_overround


def devig_power(prob_a: float, prob_b: float, tol: float = 1e-12) -> tuple[float, float]:
    """Power de-vig: find the exponent k with prob_a**k + prob_b**k == 1, by bisection.
    Shrinks longshots more than favourites, unlike the proportional method.
    """
    lo, hi = 0.5, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if prob_a ** mid + prob_b ** mid > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = (lo + hi) / 2.0
    return prob_a ** k, prob_b ** k


if __name__ == "__main__":
    games = load_games("data/games.csv")
    priced = run_walkforward(games, price_season=2025)

    priced["market_implied_home_raw"] = priced["home_moneyline"].apply(moneyline_to_implied_prob)
    priced["market_implied_away_raw"] = priced["away_moneyline"].apply(moneyline_to_implied_prob)
    devigged = priced.apply(
        lambda r: devig_two_way(r["market_implied_home_raw"], r["market_implied_away_raw"]),
        axis=1,
    )
    priced["market_prob_home"] = [d[0] for d in devigged]
    priced["market_prob_away"] = [d[1] for d in devigged]
    priced["market_vig"] = (
        priced["market_implied_home_raw"] + priced["market_implied_away_raw"] - 1.0
    )

    priced.to_csv("output/predictions_2025.csv", index=False)
    print(f"Priced {len(priced)} games from the 2025 regular season.")
    print(f"Mean market vig on the moneyline: {priced['market_vig'].mean()*100:.2f}%")
    print("Wrote output/predictions_2025.csv")
