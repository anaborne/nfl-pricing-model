# Walk-Forward NFL Elo, Graded Against the Market Line

A from-scratch NFL win-probability model, graded against the market's own moneyline for
every 2025 game. It puts a number on how much worse a simple, publicly documented pricing
model is than the market it would have to beat to be worth trading on.

This is a backtest. It is not a live prediction feed. Every game graded here already
happened, and the model's probability for each one was computed using only information
available strictly before that game (final scores and lines from earlier games, never
anything from the game being priced or anything later). No number in this repo was tuned
after seeing how it scored against 2025. The constants are named at the top of
`src/elo_model.py`, they come from public literature, and `run_walkforward` prices a game
before folding that game's result into the ratings.

## What it does

1. Rates all 32 NFL teams with an Elo-style system, walking forward game-by-game
   through the 2002-2024 regular seasons as warmup (postseason games are not used
   anywhere), then continuing through the full 2025 regular season (272 games) while
   recording each game's pre-game prediction.
2. Converts every game's recorded moneyline into a de-vigged market-implied
   probability, for an apples-to-apples comparison. The raw moneyline overstates
   both sides' odds by the bookmaker's margin, averaging 4.28% here, so the margin has to
   be removed before comparing to a model that has no vig of its own.
3. Grades both (plus a naive "home teams win X% of the time" baseline) with Brier score,
   log loss, a full calibration/reliability diagram, a paired significance test, and
   several cuts of where the loss actually comes from.

## Methodology

### Rating system

Standard Elo, modeled on the publicly documented "538-style" NFL Elo family (see
FiveThirtyEight's public NFL Elo methodology writeups, and the several open-source
reimplementations that followed, such as `nfelo`). It is not a novel formula. Constants
used, chosen once from that public literature and never adjusted after seeing a result:

- `K = 20`
- Home-field advantage: `+48` Elo points added to the home team before computing a
  win probability, and `0` where nflverse marks the game `location == Neutral`, which
  in the 2025 regular season is the seven internationals
- Margin-of-victory dampener: `ln(|margin| + 1) × 2.2 / (0.001 × elo_gap + 2.2)`. A
  blowout moves ratings more than a squeaker, but the effect shrinks the more the two
  teams' ratings already differed (beating a team you were heavily favored against
  by a lot is less informative than doing it as the underdog)
- Season carryover: at the start of a new season each team's rating regresses 1/3 of the
  way back to 1500
- Relocated franchises (Raiders OAK→LV 2020, Chargers SD→LAC 2017, Rams STL→LA 2016) keep
  one continuous rating under their current abbreviation, so the rating follows the
  organization through a move

`src/sensitivity.py` re-runs the whole walk-forward under a grid of alternatives to `K`,
home-field advantage and season carryover, plus two alternatives to the de-vig method.
The margin-of-victory dampener and the franchise-alias rule are not swept. See [Did the constants flatter the model?](#did-the-constants-flatter-the-model)
below.

### Walk-forward pricing

The model never sees the outcome of a game before pricing it. 2002-2024 (5,951
regular-season games) is pure warmup, used only to build ratings that are hot by the start
of 2025, and only 2025 games are priced and graded. The naive baseline is held to the same
rule. It predicts the 2002-2024 home-win rate (0.5607) and never 2025's own realized rate,
so even the weakest benchmark in the table is free of lookahead.

### Data

[nflverse/nfldata](https://github.com/nflverse/nfldata), a public, community-maintained
dataset (`data/games.csv`) carrying final scores and the spread for every NFL game back
to 1999, and the recorded moneyline from 2006 onward (complete from 2010 apart from a
single 2017 game; all 272 of the 2025 games graded here have one). One caveat is worth
stating. nflverse documents these fields only as "Odds for away/home team to win the game"
and does not specify whether they are opening or closing quotes, so this repo calls them
the market's recorded line and never the close. If they are pre-close quotes, the true gap
against a real closing line is wider than what this repo reports. A snapshot is vendored in
`data/games.csv` so this repo runs standalone without depending on the source staying
live, and `data/SOURCE.txt` records exactly when and from where it was pulled and repeats
this caveat.

## Reproduce it

```
pip install -r requirements.txt
python3 src/elo_model.py     # walks 2002-2025, writes output/predictions_2025.csv
python3 src/evaluate.py      # grades it, writes output/calibration_plot.png + CSVs
python3 src/sensitivity.py   # re-runs under varied constants, writes output/sensitivity.csv
```

## Results (2025 NFL regular season, n=272)

Realized home win rate: 53.7%.

| | Brier score | Log loss |
|---|---|---|
| Model (walk-forward Elo) | 0.2239 | 0.6380 |
| Market (de-vigged moneyline) | 0.2116 | 0.6082 |
| Naive (2002-2024 home-win rate) | 0.2492 | 0.6916 |

Lower is better for both metrics. The honest headline is that the market beat this model,
by about 0.012 Brier. Both clear the naive baseline comfortably, and the model's
calibration curve tracks the diagonal reasonably well for n=272, but a two-input Elo
system built from box scores alone does not out-price a market that prices in injuries,
weather, line movement, and everything else the public knows. That is the expected result.
Beating the market's own line with a textbook Elo formula and no proprietary information
would itself be the finding worth doubting. What this artifact demonstrates is a real
pricing methodology, run walk-forward with no lookahead, graded against the market's own
number, with the loss reported.

The model is mildly overconfident, and that accounts for very little of the gap. Fitting
`y ~ sigmoid(a + b·logit(p))` to each forecaster gives a recalibration slope of 0.877
for the model and 0.985 for the market (`output/recalibration.csv`). A slope of 1 is
perfect calibration in the logit sense, and below 1 means the probabilities are too extreme
for how often they are right. The market's 0.985 is as close to 1 as this sample can show.
The model's 0.877 says its 80% picks should have been more like 76% picks. Correcting that
is the most generous fix available, since the coefficients are fit in sample on the same
272 games, and it moves the model's Brier from 0.2239 only to 0.2229, which closes 8% of
the gap. Recalibrating both forecasters leaves 93% of the gap standing. What is left is
resolution. The market's probabilities separate winners from losers better, AUC 0.720
against 0.687 (`output/discrimination.csv`), and that is what the disagreement table below
shows game by game.

See `output/calibration_plot.png` for the full reliability diagram (model vs. market,
both plotted against the realized frequency in each fixed-width 0.1 probability bin,
with both recalibration slopes in the legend). The bins are equal width. Each holds
however many games fell into it, and the marker at the top of the range stands on very
few.

### Is the gap real, or is n=272 too small?

The two forecasters price the same 272 games, so the losses are paired and the paired
test is the right one. It differences out how hard each game was. On per-game Brier
loss, model minus market (`output/paired_test.csv`):

| Metric | Mean difference | SE | t | 95% CI |
|---|---|---|---|---|
| Brier | +0.01231 | 0.00517 | 2.38 | [+0.00217, +0.02245] |
| Log loss | +0.02976 | 0.01168 | 2.55 | [+0.00688, +0.05265] |

Both intervals exclude zero. One season is still one season, and the lower bound is
small, but the market's edge here is not a sampling artifact.

### Where the model disagrees with the market

If a model has a tradeable edge, it shows up where the model and the market disagree,
because that is where a bettor would actually act. Agreement is worth nothing. Slicing on
`|model_prob − market_prob|` (`output/disagreement.csv`):

| Disagreement > | n | Model Brier | Market Brier | Gap | Model hit rate | Market hit rate |
|---|---|---|---|---|---|---|
| 0.00 (all games) | 272 | 0.2239 | 0.2116 | +0.0123 | 64.3% | 65.8% |
| 0.05 | 138 | 0.2275 | 0.2072 | +0.0203 | 63.0% | 65.2% |
| 0.10 | 57 | 0.2614 | 0.2040 | +0.0574 | 50.9% | 64.9% |
| 0.15 | 28 | 0.2869 | 0.2000 | +0.0869 | 46.4% | 67.9% |
| 0.20 | 14 | 0.2929 | 0.1825 | +0.1104 | 42.9% | 71.4% |

This is the sharpest version of the repo's own thesis. The gap widens monotonically as
disagreement grows, more than eightfold from the full sample to the most-contested 14
games, and the model's directional hit rate falls below a coin flip while the market's
climbs to 71%. When this model most confidently departs from the market, it is most
reliably wrong. That is the opposite of a tradeable signal, and it is exactly what you
would expect from a rating system whose information set is a strict subset of the market's.

### When the gap opens up

Splitting the season (`output/by_week_bucket.csv`):

| Weeks | n | Model Brier | Market Brier | Gap |
|---|---|---|---|---|
| 1-4 | 64 | 0.2021 | 0.1931 | +0.0089 |
| 5-17 | 192 | 0.2309 | 0.2185 | +0.0124 |
| 18 | 16 | 0.2278 | 0.2031 | +0.0247 |

Week 18 is where the gap is widest, at roughly double the full-season figure on only 16
games. Weeks 17 and 18 are when playoff seeds get clinched and teams rest starters, and a
rating built from box scores has no way to know that the team it rates at 1650 is about to
play its backups. The biggest single miss below is the same case.

### Did the constants flatter the model?

`src/sensitivity.py` re-runs the entire walk-forward under a grid of alternatives and
writes `output/sensitivity.csv`. Model Brier at each setting, with everything else held
at the published values:

| Sweep | Values | Model Brier |
|---|---|---|
| `K` | 12 / 16 / 20 / 24 / 30 | 0.2262 / 0.2247 / 0.2239 / 0.2237 / 0.2241 |
| Home-field advantage | 0 / 25 / 48 / 55 / 65 | 0.2234 / 0.2227 / 0.2239 / 0.2246 / 0.2259 |
| Season regression | 0 / 0.25 / 1/3 / 0.5 | 0.2317 / 0.2251 / 0.2239 / 0.2227 |

The published setting in each row is `K` = 20, home-field advantage = 48, and season
regression = 1/3, which score 0.2239 in all three sweeps. Swapping the de-vig method
changes the market's probabilities and leaves the model's untouched. Proportional
(published) leaves a gap of 0.0123, power 0.0120, additive 0.0121.

Two things are worth reading off this table. First, the published configuration is not the
best cell in it. K=24, HFA=0 or 25, and a 0.5 season regression each score marginally better,
which is what fixing constants from public literature, with no fit to this test set, looks
like. Second, the best configuration anywhere in this sweep still loses to the market by
about 0.011 Brier. No choice of constants on this grid turns the result around, so the
conclusion does not depend on the ones that were picked.

### The ten games the model was most wrong about

(Ranked by how much probability the model put on the outcome that actually happened,
lowest first. Market column shown for comparison. Full table with the model-minus-market
confidence gap in `output/biggest_misses.csv`.)

- Week 9: CAR 16 at GB 13, model gave GB 84% (market 87%)
- Week 5: NE 23 at BUF 20, model gave BUF 84% (market 77%)
- Week 18: WAS 24 at PHI 17, model gave PHI 83% (market 62%)
- Week 16: KC 9 at TEN 26, model gave KC 82% (market 62%)
- Week 6: PHI 17 at NYG 34, model gave PHI 81% (market 77%)
- Week 14: NO 24 at TB 20, model gave TB 81% (market 78%)
- Week 13: CIN 32 at BAL 14, model gave BAL 79% (market 76%)
- Week 13: LA 28 at CAR 31, model gave LA 78% (market 82%)
- Week 5: TEN 22 at ARI 21, model gave ARI 78% (market 78%)
- Week 2: ATL 22 at MIN 6, model gave MIN 77% (market 62%)

In seven of these ten the model was more confident than the market in the side that
lost, and in three (Week 18 WAS@PHI, Week 16 KC@TEN, Week 2 ATL@MIN) it was more
confident by 15 points or more. Week 18 is the clearest case. Philadelphia had its seed
settled and rested Hurts along with five other starters, which the market priced at 62%
and a rating system built from box scores priced at 83%. That gap shows the width of the
model's information set, and it is not noise. The remaining seven are unremarkable, in
that a well-calibrated 80% favorite is supposed to lose about one time in five, and a
calibration curve that never produced a wrong 80% pick would itself be evidence of
miscalibration, since an 80% pick that never loses is an 80% pick priced wrong.

### How the one tie is graded

2025 had exactly one tie: GB 40 at DAL 40 in week 4. The graded event throughout this repo
is literally "did the home team win", so a tie is a home non-win. That is the published
treatment. `src/evaluate.py` prints all three defensible treatments
(`output/tie_treatment.csv`):

| Treatment | n | Model Brier | Market Brier |
|---|---|---|---|
| As published (tie = home non-win) | 272 | 0.2239 | 0.2116 |
| Tie scored at 0.5 | 272 | 0.2235 | 0.2116 |
| Tie excluded | 271 | 0.2243 | 0.2121 |

The ordering does not change under any of the three.

## Limitations, stated plainly

- Inputs are box scores only, with no injuries, weather, referee assignments, or line
  movement. The market prices all of those. This model prices none of them.
- No player-level or roster continuity signal (a team's Elo doesn't know it lost its
  starting QB, or that six starters are being rested in week 18).
- Constants are public-domain defaults with no fit to this dataset, which is a deliberate
  honesty choice (see Methodology). It also means there was no attempt to squeeze out the
  last few points of accuracy the way a production pricing model would.
- The benchmark is the market's recorded moneyline, which nflverse does not document as
  opening or closing. See the Data section. If these are pre-close quotes, the true gap
  against a real closing line is wider than reported here.
- This is a single-season backtest (n=272). The paired 95% CI on the Brier difference is
  [+0.00217, +0.02245]. It excludes zero, its lower bound is small, and one season
  cannot separate a 0.012 gap from a 0.002 one.

## What this is not

This is not a live trading system, and it is not a claim of a verified P&L track record.
It grades a pricing methodology against history, including where it lost.

## License

MIT, see [LICENSE](LICENSE). The vendored `data/games.csv` snapshot comes from
[nflverse/nfldata](https://github.com/nflverse/nfldata), which is public and
community-maintained. It publishes no LICENSE file of its own, so no license is claimed
for it here on its behalf.
