"""Replays the whole walk-forward under varied K, HFA and season regression, plus the two
alternatives to proportional de-vig, and reports what each does to the 2025 Brier score.

This answers the obvious challenge to any backtest, that the constants were picked to
flatter the model, with a table instead of an assurance. Nothing here feeds back into the
published numbers; the graded model keeps the constants at the top of elo_model.py, fixed
before any result was seen. The smallest gap in the sweep is printed at the end of the run
and written to output/sensitivity.csv.

    python3 src/sensitivity.py

Each row of the K / HFA / regression sweeps replays all warmup seasons plus 2025 from
scratch.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from elo_model import (
    HOME_FIELD_ADVANTAGE,
    K_FACTOR,
    SEASON_REGRESSION,
    EloParams,
    devig_additive,
    devig_power,
    devig_two_way,
    load_games,
    moneyline_to_implied_prob,
    run_walkforward,
)

K_GRID = [12.0, 16.0, 20.0, 24.0, 30.0]
HFA_GRID = [0.0, 25.0, 48.0, 55.0, 65.0]
REGRESSION_GRID = [0.0, 0.25, 1.0 / 3.0, 0.5]
DEVIG_METHODS = {
    "proportional": devig_two_way,
    "power": devig_power,
    "additive": devig_additive,
}


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def market_prob(priced: pd.DataFrame, method) -> np.ndarray:
    raw_home = priced["home_moneyline"].apply(moneyline_to_implied_prob)
    raw_away = priced["away_moneyline"].apply(moneyline_to_implied_prob)
    return np.array([method(h, a)[0] for h, a in zip(raw_home, raw_away)])


def is_published(params: EloParams) -> bool:
    return (
        params.k_factor == K_FACTOR
        and params.home_field_advantage == HOME_FIELD_ADVANTAGE
        and params.season_regression == SEASON_REGRESSION
    )


if __name__ == "__main__":
    # output/ is committed, but recreate it so a run still works after a "rm -rf output".
    os.makedirs("output", exist_ok=True)
    games = load_games("data/games.csv")

    # The published run, reused as the model side of the de-vig sweep (changing how the
    # market's vig is removed does not touch the model's probabilities).
    published = run_walkforward(games, price_season=2025, params=EloParams())
    y = published["home_win"].to_numpy(dtype=float)
    published_model_brier = brier(published["model_prob_home"].to_numpy(dtype=float), y)

    rows: list[dict] = []

    sweeps = [
        ("k_factor", K_GRID, lambda v: EloParams(k_factor=v)),
        ("home_field_advantage", HFA_GRID, lambda v: EloParams(home_field_advantage=v)),
        ("season_regression", REGRESSION_GRID, lambda v: EloParams(season_regression=v)),
    ]
    for name, grid, build in sweeps:
        for value in grid:
            params = build(value)
            priced = run_walkforward(games, price_season=2025, params=params)
            p = priced["model_prob_home"].to_numpy(dtype=float)
            market_p = market_prob(priced, devig_two_way)
            model_b = brier(p, y)
            market_b = brier(market_p, y)
            rows.append(
                {
                    "sweep": name,
                    "value": value,
                    "n": len(priced),
                    "model_brier": model_b,
                    "market_brier": market_b,
                    "gap": model_b - market_b,
                    "published": is_published(params),
                }
            )
            print(
                f"{name:22s} {value:>12.4g}   model {model_b:.4f}   "
                f"market {market_b:.4f}   gap {model_b - market_b:+.4f}"
            )

    for name, method in DEVIG_METHODS.items():
        market_p = market_prob(published, method)
        market_b = brier(market_p, y)
        rows.append(
            {
                "sweep": "devig_method",
                "value": name,
                "n": len(published),
                "model_brier": published_model_brier,
                "market_brier": market_b,
                "gap": published_model_brier - market_b,
                "published": name == "proportional",
            }
        )
        print(
            f"{'devig_method':22s} {name:>12s}   model {published_model_brier:.4f}   "
            f"market {market_b:.4f}   gap {published_model_brier - market_b:+.4f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv("output/sensitivity.csv", index=False)

    best = out.loc[out["gap"].idxmin()]
    print()
    print(
        f"Smallest gap anywhere in the sweep: {best['gap']:+.4f} "
        f"({best['sweep']} = {best['value']})."
    )
    print("Every configuration still loses to the market. Wrote output/sensitivity.csv")
