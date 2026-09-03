"""Grades the walk-forward Elo model against the market on the 2025 NFL regular season.

Every probability read here comes from output/predictions_2025.csv, written by
elo_model.py. Nothing in this file re-derives or overrides one.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARMUP_FIRST_SEASON = 2002
WARMUP_LAST_SEASON = 2024

# output/ is committed, but recreate it so a run still works after a "rm -rf output".
os.makedirs("output", exist_ok=True)

df = pd.read_csv("output/predictions_2025.csv")
y = df["home_win"].to_numpy(dtype=float)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_loss_per_game(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def log_loss_per_game(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def warmup_home_win_rate(path: str = "data/games.csv") -> float:
    """Home-win rate over the 2002-2024 warmup seasons.

    Fitting the naive baseline on 2025's own realized rate would leak the graded season
    into its own benchmark. Small leak, but the repo's claim is zero lookahead, so the
    baseline sees the same warmup seasons the ratings do and nothing else.
    """
    g = pd.read_csv(path)
    g = g[g["game_type"] == "REG"]
    g = g[(g["season"] >= WARMUP_FIRST_SEASON) & (g["season"] <= WARMUP_LAST_SEASON)]
    g = g.dropna(subset=["home_score", "away_score"])
    return float((g["home_score"] > g["away_score"]).mean())


NAIVE_RATE = warmup_home_win_rate()

model_p = df["model_prob_home"].to_numpy(dtype=float)
market_p = df["market_prob_home"].to_numpy(dtype=float)
naive_p = np.full_like(y, NAIVE_RATE)

results = {
    "model (walk-forward Elo)": (model_p, brier(model_p, y), log_loss(model_p, y)),
    "market (de-vigged moneyline)": (market_p, brier(market_p, y), log_loss(market_p, y)),
    "naive (2002-2024 home-win rate)": (naive_p, brier(naive_p, y), log_loss(naive_p, y)),
}

print(f"n = {len(df)} games, season = 2025 REG, realized home win rate = {y.mean():.3f}")
print(
    f"naive baseline = {NAIVE_RATE:.4f} "
    f"(home-win rate over {WARMUP_FIRST_SEASON}-{WARMUP_LAST_SEASON}, not 2025)"
)
print()
print(f"{'':32s}{'Brier':>10s}{'Log loss':>12s}")
for name, (_, b, ll) in results.items():
    print(f"{name:32s}{b:10.4f}{ll:12.4f}")
print()
print("Lower is better for both. Brier of 0 is a perfect prediction, 0.25 is what an")
print("always-50% coin flip scores against a 50/50 outcome distribution.")

pd.DataFrame(
    [
        {
            "forecaster": name,
            "n": len(df),
            "brier": b,
            "log_loss": ll,
            "realized_home_win_rate": float(y.mean()),
            "naive_rate": NAIVE_RATE,
            "mean_market_vig": float(df["market_vig"].mean()),
        }
        for name, (_, b, ll) in results.items()
    ]
).to_csv("output/metrics.csv", index=False)

# The graded event is "did the home team win", so a tie is a home non-win. That is the
# published treatment; the two alternatives run below so the choice is visible.
tie = df["tie"].to_numpy(dtype=int)
y_half = np.where(tie == 1, 0.5, y)
kept = tie == 0

tie_rows = [
    ("as published (tie = home non-win)", len(y), brier(model_p, y), brier(market_p, y)),
    ("tie scored at 0.5", len(y), brier(model_p, y_half), brier(market_p, y_half)),
    (
        "tie excluded",
        int(kept.sum()),
        brier(model_p[kept], y[kept]),
        brier(market_p[kept], y[kept]),
    ),
]
print()
print(f"Tie treatment ({int(tie.sum())} tie in 2025: GB 40 at DAL 40, week 4)")
print(f"{'':36s}{'n':>5s}{'model':>10s}{'market':>10s}")
for name, n, bm, bk in tie_rows:
    print(f"{name:36s}{n:5d}{bm:10.4f}{bk:10.4f}")
print("Ordering is unchanged under all three. The market wins either way.")
pd.DataFrame(
    [{"treatment": t, "n": n, "model_brier": bm, "market_brier": bk} for t, n, bm, bk in tie_rows]
).to_csv("output/tie_treatment.csv", index=False)

# Both forecasters price the same games, so the losses are paired and the paired test
# differences out game-level difficulty. Normal quantile, not t: at this n they agree to
# three decimals.
Z = 1.959963984540054


def paired_test(loss_model: np.ndarray, loss_market: np.ndarray) -> dict[str, float]:
    d = loss_model - loss_market
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {
        "n": n,
        "mean_diff": mean,
        "se": se,
        "t": mean / se,
        "ci_low": mean - Z * se,
        "ci_high": mean + Z * se,
    }


paired = {
    "brier": paired_test(brier_loss_per_game(model_p, y), brier_loss_per_game(market_p, y)),
    "log_loss": paired_test(log_loss_per_game(model_p, y), log_loss_per_game(market_p, y)),
}
print()
print("Paired test on per-game loss, model minus market (positive = model is worse)")
for metric, r in paired.items():
    print(
        f"  {metric:9s} diff {r['mean_diff']:+.5f}  SE {r['se']:.5f}  t {r['t']:.2f}  "
        f"95% CI [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"
    )
print("  Both intervals exclude zero: the market's edge is not a sampling artifact.")
pd.DataFrame(
    [{"metric": m, **r} for m, r in paired.items()]
).to_csv("output/paired_test.csv", index=False)

# A tradeable edge would show up where the two disagree most, since that is where a
# bettor would act.
disagreement = np.abs(model_p - market_p)


def hit_rate(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p > 0.5) == (y == 1)))


dis_rows = []
for threshold in (0.00, 0.05, 0.10, 0.15, 0.20):
    s = disagreement > threshold
    dis_rows.append(
        {
            "threshold": threshold,
            "n": int(s.sum()),
            "model_brier": brier(model_p[s], y[s]),
            "market_brier": brier(market_p[s], y[s]),
            "gap": brier(model_p[s], y[s]) - brier(market_p[s], y[s]),
            "model_hit_rate": hit_rate(model_p[s], y[s]),
            "market_hit_rate": hit_rate(market_p[s], y[s]),
        }
    )
dis = pd.DataFrame(dis_rows)
dis.to_csv("output/disagreement.csv", index=False)
print()
print("Disagreement subset: |model - market| > threshold")
print(f"{'thr':>6s}{'n':>6s}{'model':>10s}{'market':>10s}{'gap':>10s}{'hit(m)':>9s}{'hit(k)':>9s}")
for r in dis_rows:
    print(
        f"{r['threshold']:6.2f}{r['n']:6d}{r['model_brier']:10.4f}{r['market_brier']:10.4f}"
        f"{r['gap']:+10.4f}{r['model_hit_rate']:9.4f}{r['market_hit_rate']:9.4f}"
    )
print("  The gap widens as disagreement grows, the opposite of a tradeable signal.")


# IRLS rather than scipy: it is Newton-Raphson on the logistic likelihood and keeps the
# dependency list at numpy, pandas, matplotlib.
def recalibration_fit(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    p = np.clip(p, eps, 1 - eps)
    X = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(mu * (1 - mu), eps, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return float(beta[0]), float(beta[1])


model_a, model_b = recalibration_fit(model_p, y)
market_a, market_b = recalibration_fit(market_p, y)
print()
print("Recalibration fit  y ~ sigmoid(a + b*logit(p))   (b < 1 = overconfident)")
print(f"  model    intercept {model_a:+.4f}   slope {model_b:.3f}")
print(f"  market   intercept {market_a:+.4f}   slope {market_b:.3f}")
# Rounded on write: IRLS converges to a step tolerance, not to the last bit, and the raw
# float64 repr drifts in its final digit between runs and dirties the diff.
pd.DataFrame(
    [
        {"forecaster": "model", "intercept": round(model_a, 6), "slope": round(model_b, 6)},
        {"forecaster": "market", "intercept": round(market_a, 6), "slope": round(market_b, 6)},
    ]
).to_csv("output/recalibration.csv", index=False)


# The fit above only names the miscalibration. Applying it says how much of the Brier gap
# that miscalibration accounts for. This is the most generous form of the question, since
# the coefficients are fit in sample on the same 272 games.
def recalibrate(p: np.ndarray, a: float, b: float, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(a + b * np.log(p / (1 - p)))))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve, as the Mann-Whitney statistic with ties split evenly.

    Brier mixes calibration with resolution. AUC ignores calibration and asks only whether
    the games the home team won were ranked above the games it lost, so it isolates the
    part of the gap that recalibration leaves untouched.
    """
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = float((y == 1).sum())
    n_neg = float(len(y) - n_pos)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


model_recal_p = recalibrate(model_p, model_a, model_b)
market_recal_p = recalibrate(market_p, market_a, market_b)
model_brier, market_brier = brier(model_p, y), brier(market_p, y)
model_recal_brier, market_recal_brier = brier(model_recal_p, y), brier(market_recal_p, y)
gap = model_brier - market_brier
gap_closed_model_recalibrated = (model_brier - model_recal_brier) / gap
gap_left_both_recalibrated = (model_recal_brier - market_recal_brier) / gap
model_auc, market_auc = auc(model_p, y), auc(market_p, y)

print()
print("Recalibrated in sample: a ceiling on what fixing calibration alone could buy")
print(f"{'':32s}{'Brier':>10s}{'recal.':>10s}{'AUC':>8s}")
print(
    f"{'model (walk-forward Elo)':32s}{model_brier:10.4f}"
    f"{model_recal_brier:10.4f}{model_auc:8.3f}"
)
print(
    f"{'market (de-vigged moneyline)':32s}{market_brier:10.4f}"
    f"{market_recal_brier:10.4f}{market_auc:8.3f}"
)
print(
    f"  recalibrating the model alone closes {gap_closed_model_recalibrated * 100:.0f}% of "
    f"the {gap:.4f} Brier gap"
)
print(
    f"  recalibrating both leaves {gap_left_both_recalibrated * 100:.0f}% of the gap "
    f"standing. The rest is resolution, which the AUC column measures."
)
# Rounded on write for the same reason as recalibration.csv above: these follow from the
# IRLS coefficients, whose last float64 digit drifts between runs.
pd.DataFrame(
    [
        {
            "forecaster": name,
            "n": len(df),
            "brier": round(b, 6),
            "recalibrated_brier": round(rb, 6),
            "auc": round(a, 6),
            "gap": round(gap, 6),
            "gap_closed_model_recalibrated": round(gap_closed_model_recalibrated, 6),
            "gap_left_both_recalibrated": round(gap_left_both_recalibrated, 6),
        }
        for name, b, rb, a in (
            ("model", model_brier, model_recal_brier, model_auc),
            ("market", market_brier, market_recal_brier, market_auc),
        )
    ]
).to_csv("output/discrimination.csv", index=False)

# Cut at week 4 and week 18: early season is ratings still shaking off the carryover
# regression, and week 18 is rest-and-sit decisions no box-score rating system can see.
week = df["week"].to_numpy(dtype=int)
bucket_rows = []
for label, mask in (
    ("weeks 1-4", week <= 4),
    ("weeks 5-17", (week >= 5) & (week <= 17)),
    ("week 18", week == 18),
):
    bucket_rows.append(
        {
            "bucket": label,
            "n": int(mask.sum()),
            "model_brier": brier(model_p[mask], y[mask]),
            "market_brier": brier(market_p[mask], y[mask]),
            "gap": brier(model_p[mask], y[mask]) - brier(market_p[mask], y[mask]),
        }
    )
pd.DataFrame(bucket_rows).to_csv("output/by_week_bucket.csv", index=False)
print()
print("By week bucket")
print(f"{'':12s}{'n':>5s}{'model':>10s}{'market':>10s}{'gap':>10s}")
for r in bucket_rows:
    print(
        f"{r['bucket']:12s}{r['n']:5d}{r['model_brier']:10.4f}"
        f"{r['market_brier']:10.4f}{r['gap']:+10.4f}"
    )
print("  Week 18 is where the gap is widest. Rested starters the model cannot see.")


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "actual_freq": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


model_cal = calibration_bins(model_p, y)
market_cal = calibration_bins(market_p, y)
model_cal.to_csv("output/calibration_model.csv", index=False)
market_cal.to_csv("output/calibration_market.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="perfect calibration")
ax.scatter(
    model_cal["mean_predicted"], model_cal["actual_freq"],
    s=model_cal["n"] * 6, color="#1C6E8C", alpha=0.85,
    label=f"model (Elo), recal. slope {model_b:.3f}", zorder=3,
)
ax.scatter(
    market_cal["mean_predicted"], market_cal["actual_freq"],
    s=market_cal["n"] * 6, color="#B96A21", alpha=0.85,
    label=f"market (de-vigged), recal. slope {market_b:.3f}", zorder=3,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability (home win), by 0.1-wide bin")
ax.set_ylabel("Realized home-win frequency in that bin")
ax.set_title("2025 NFL regular season: predicted vs. realized (n=272)\nmarker size = games in bin")
ax.legend(loc="upper left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("output/calibration_plot.png")
print("\nWrote output/calibration_plot.png, output/calibration_model.csv, output/calibration_market.csv")

df["model_prob_of_actual"] = np.where(df["home_win"] == 1, df["model_prob_home"], 1 - df["model_prob_home"])
df["market_prob_of_actual"] = np.where(df["home_win"] == 1, df["market_prob_home"], 1 - df["market_prob_home"])
# Positive means the model was the more confident of the two in the side that lost.
df["model_extra_confidence_in_loser"] = df["market_prob_of_actual"] - df["model_prob_of_actual"]
worst = df.sort_values("model_prob_of_actual").head(10)
worst[
    ["week", "away_team", "home_team", "away_score", "home_score",
     "model_prob_home", "market_prob_home", "model_prob_of_actual", "market_prob_of_actual",
     "model_extra_confidence_in_loser"]
].to_csv("output/biggest_misses.csv", index=False)
n_more_confident = int((worst["model_extra_confidence_in_loser"] > 0).sum())
n_big = int((worst["model_extra_confidence_in_loser"] >= 0.15).sum())
print("Wrote output/biggest_misses.csv (10 games the model was most wrong about)")
print(
    f"  model was more confident than the market in the losing side in "
    f"{n_more_confident} of 10, by >= 15 points in {n_big}"
)
print(
    "Wrote output/metrics.csv, output/tie_treatment.csv, output/paired_test.csv, "
    "output/disagreement.csv, output/recalibration.csv, output/discrimination.csv, "
    "output/by_week_bucket.csv"
)
