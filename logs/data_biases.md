# FPL Data Bias Analysis

**Date:** 2026-03-15
**Purpose:** Evaluate systematic biases introduced by using FPL data as a proxy for Premier League player performance, before training ML models.
**Database:** `db/fpl.db` — 242,316 GW rows, 10 seasons (2016-17 to 2025-26)

---

## Context

FPL data is used as a proxy for real Premier League performance data because it is the richest freely available dataset covering player-level statistics across 10 seasons. However, FPL is a fantasy game — its scoring system, player classifications, and coverage are designed for game engagement, not athletic measurement. This document identifies and quantifies the biases this introduces into any ML model trained on the data.

---

## Summary

| # | Bias | Severity | ML Impact |
|---|------|:--------:|-----------|
| 1 | Scoring system design | HIGH | Attacking players artificially inflated vs defenders |
| 2 | Team strength confounding | **CRITICAL** | r=-0.90 — team quality explains 81% of CS variance |
| 3 | Home advantage | HIGH | Persistent +12–20% points boost; away fixtures underestimated |
| 4 | Temporal drift & era incompatibility | HIGH | -26% points deflation over 10 seasons; xG only from 2022-23 |
| 5 | Sparsity & survivorship | **CRITICAL** | 75% of data is elite 30+ GW starters; 28% have zero minutes |
| 6 | Price as lagging indicator | MEDIUM | start_cost r=0.69 with points but rises after performance |
| 7 | Positional classification | MEDIUM | MID category spans 7x goals/90 range — too heterogeneous |
| 8 | Manager mode contamination | LOW | 322 rows in 2024-25 need filtering |
| 9 | Fixture difficulty (missing feature) | **CRITICAL** | Bottom-6 opponents give 30–50% more points — not in dataset |
| 10 | Transfer activity leakage | MEDIUM | transfers_in spikes 35x after best GW — reactive, not predictive |

---

## Bias 1 — Scoring System Design

**Severity: HIGH**

FPL points are not a neutral performance metric. The scoring rules are designed for game engagement:

- Goals: FWD=4pts, MID=5pts, DEF/GK=6pts (inverted from real-world value)
- Clean sheets: DEF/GK=4pts, MID=1pt, FWD=0pts
- Assists=3pts, Bonus=1–3pts, Yellow=-1pt, Red=-3pts

### Average total_points per season by position (dim_player_season)

| Position | Avg Season Points | Goals % of pts | Clean Sheet % | Bonus % |
|----------|:-----------------:|:--------------:|:-------------:|:-------:|
| GK | 66.5 | 0.0% | 30.0% | 7.6% |
| FWD | 60.2 | 38.5% | 23.1% | 11.4% |
| MID | 57.9 | 19.4% | 28.5% | 6.7% |
| DEF | 53.2 | 6.6% | 32.8% | 7.3% |

**Findings:**
- GKs outscore FWDs on average — driven by clean sheet points on good teams, not outfield contribution.
- Forwards earn 38.5% of their points from goals alone; defenders earn 32.8% from clean sheets alone.
- Bonus points are heavily concentrated: the top ~1.8% of bonus earners (Salah: 55, Kane: 48) accumulate disproportionately.
- Cross-position comparison using raw FPL points will misrepresent relative contribution.

**ML implication:** A position-agnostic model will learn that "GKs are as valuable as FWDs" — a game artefact, not a real-world signal. **Train position-specific models** or explicitly encode position as an interaction term.

---

## Bias 2 — Team Strength Confounding

**Severity: CRITICAL**

Defenders and goalkeepers earn clean sheet points based on their team's defensive record, not their individual defensive contribution. A world-class defender on a poor team will score far fewer CS points than an average defender on an elite team.

### Clean sheets per 90 minutes by team (2023-24)

| Tier | Team | CS/90 |
|------|------|:-----:|
| Top | Arsenal | 0.488 |
| Top | Man City | 0.339 |
| Top | Liverpool | 0.321 |
| … | … | … |
| Bottom | Sheffield Utd | 0.033 |
| Bottom | Burnley | 0.047 |
| Bottom | Luton | 0.063 |

**Range:** 0.455 CS/90 between best and worst teams — a **14.8x difference**.

### Correlation: team goals conceded vs defender CS rate

**r = -0.900** (Pearson)
Team goals conceded explains **81% of the variance** in clean sheet rate for that team's defenders and GKs.

**Findings:**
- Individual defensive skill is almost entirely masked by team quality in FPL CS data.
- A model trained without a team quality control variable will attribute team strength to individual player quality.
- This is the most structurally dangerous bias for any model predicting defensive player performance.

**ML implication:** Normalise defensive stats relative to team average. Create a `team_goals_conceded_season` control variable. Consider `player_cs_rate / team_cs_rate` as a normalised feature rather than raw clean sheet counts.

---

## Bias 3 — Home Advantage

**Severity: HIGH**

PL players consistently score more FPL points at home. This effect is real but must be explicitly modelled — a model without a `was_home` feature will produce systematically biased predictions.

### Average points per GW appearance — home vs away

| Position | Home | Away | Difference |
|----------|:----:|:----:|:----------:|
| GK | 3.66 | 3.40 | +7.8% |
| DEF | 3.06 | 2.54 | **+20.5%** |
| MID | 2.98 | 2.70 | +10.3% |
| FWD | 3.25 | 2.92 | +11.2% |

- **Defenders show the largest home advantage** (+20.5%), driven by higher clean sheet rates at home.
- Effect is consistent across all 10 seasons (median +12–15%).
- The COVID 2019-20 bubble season (played at neutral venues) shows the lowest home advantage (+0.7%), validating that the effect is genuine, not a data artefact.

**ML implication:** `was_home` must be a feature in any GW-level prediction model. Home/away × position interaction terms are also justified by the data.

---

## Bias 4 — Temporal Drift & Era Incompatibility

**Severity: HIGH**

The FPL dataset spans 10 seasons with significant changes in both scoring (game meta-shifts, rule changes) and data availability (6 distinct schema eras).

### Average points per GW appearance by season

| Season | Avg pts/GW |
|--------|:----------:|
| 2016-17 | 1.38 |
| 2017-18 | 1.38 |
| 2018-19 | **1.42** (peak) |
| 2019-20 | 1.34 |
| 2020-21 | 1.23 |
| 2021-22 | 1.18 |
| 2022-23 | 1.13 |
| 2023-24 | **1.05** (lowest) |
| 2024-25 | 1.11 |
| 2025-26 | 1.08 (partial) |

**-26.1% decline** in average points per GW from peak (2018-19) to trough (2023-24). This reflects game meta shifts — more rotation, squad depth, managerial caution — not data quality changes.

### Feature availability by era

| Era | Seasons | Missing features |
|-----|---------|-----------------|
| Old Opta | 2016-17 to 2018-19 | No xP, no xG, no starts, no position in GW data |
| Stripped | 2019-20 | No xP, no xG, no starts, no position |
| Modern core | 2020-21 to 2021-22 | No xG, no starts |
| xG era | 2022-23 to 2023-24 | Full feature set (xG, xA, xGI, xGC, starts, xP) |
| Manager era | 2024-25 | Full + mng_* columns |
| Defensive era | 2025-26 | Full + defensive stats, no starts |

**Findings:**
- The richest 4 seasons (2022-23 to 2025-26) have xG data; the remaining 6 do not.
- Naively combining all 10 seasons as one training set introduces severe feature sparsity for 60% of the data.
- Scoring deflation means 2016-19 and 2022-25 data are not directly comparable without normalisation.

**ML implication:** Consider training era-specific models (pre-xG vs xG era) or adding an `era_id` feature and normalising points to season mean. Do not use xG-dependent features on pre-2022-23 data without imputation strategy.

---

## Bias 5 — Sparsity & Survivorship

**Severity: CRITICAL**

The dataset is dominated by elite, regular-starting players. Players who rotate, get injured, or underperform are underrepresented — both in volume and feature richness.

### Distribution of GW appearances per player-season

| GW appearances | Player-seasons | % of total |
|:--------------:|:--------------:|:----------:|
| 30+ | 5,474 | **74.6%** |
| 20–29 | 1,054 | 14.4% |
| 10–19 | 527 | 7.2% |
| 5–9 | 146 | 2.0% |
| 1–4 | 133 | 1.8% |

### Distribution of minutes per player-season

| Minutes played | Player-seasons | % |
|:--------------:|:--------------:|:-:|
| 0 (never played) | 2,027 | **27.6%** |
| 1–499 | 1,461 | 19.9% |
| 500–999 | 697 | 9.5% |
| 1000–1999 | 1,580 | 21.5% |
| 2000+ | 1,569 | 21.4% |

### Player career length

| Seasons appeared | Players | % |
|:----------------:|:-------:|:-:|
| 1 season only | 1,033 | **39.4%** |
| 2–3 seasons | 809 | 30.9% |
| 4–5 seasons | 436 | 16.7% |
| 6+ seasons | 342 | **13.1%** |

**Findings:**
- 75% of training data comes from players with 30+ GW appearances — elite, regular starters.
- 28% of player-seasons have zero minutes played (registered in FPL but never used).
- 39% of players appear in only one season — high squad turnover means limited longitudinal data for most players.
- Models trained on this data effectively learn "how do elite players perform" — not generalisable to rotation or fringe players.

**ML implication:** Filter out player-seasons with fewer than 5 GW appearances or fewer than 200 minutes before training. Stratify validation metrics by minutes bucket. Acknowledge that model performance will degrade for low-minute players.

---

## Bias 6 — Price as a Lagging Indicator

**Severity: MEDIUM**

FPL price (`start_cost`, `value`) reflects fantasy manager consensus, which is reactive to recent performance.

### Correlation: start_cost vs season total_points

**r = 0.69** (moderate positive correlation)

### Average season points by price band

| Price band | Avg season pts | Std dev |
|:----------:|:--------------:|:-------:|
| < £5.0m (< 50) | 20.5 | 32.7 |
| £5.0–6.9m (50–69) | 56.4 | 45.3 |
| £7.0–8.9m (70–89) | 93.1 | 56.4 |
| £9.0–10.9m (90–109) | 124.8 | 65.8 |
| £11.0m+ (110+) | 186.4 | 58.6 |

**Findings:**
- Price correlates with points (r=0.69), but the within-band standard deviation is nearly as large as the between-band difference — price is a noisy signal.
- `start_cost` is set before the season based on prior-season performance; it is a lagging indicator of past quality, not a leading indicator of future performance.
- Within-season `value` changes (GW-by-GW price movements) reflect FPL transfer activity, which is itself reactive (see Bias 10).

**ML implication:** `start_cost` can be used as a feature representing prior-season perceived value but should not be treated as a ground-truth quality signal. Never use within-season `value` as a predictive feature without lagging it.

---

## Bias 7 — Positional Classification

**Severity: MEDIUM**

FPL position categories are defined for squad-building purposes, not athletic role. The MID category in particular conflates radically different player types.

### Goals scored per 90 minutes, within each position

| Position | Mean goals/90 | Std dev | Range |
|----------|:-------------:|:-------:|:-----:|
| FWD | 0.379 | 0.21 | 0–1.2 |
| MID | 0.158 | 0.19 | **0.01–0.35** |
| DEF | 0.043 | 0.08 | 0–0.4 |
| GK | 0.000 | — | 0 |

**Coefficient of variation within MID: 1.127** (highest of all outfield positions)

**Findings:**
- The MID category spans a 7x range in goals/90 — from pure defensive midfielders (~0.01) to attacking wingers (~0.35).
- Wingers (e.g., Salah, Son), box-to-box midfielders, and holding midfielders are all classified identically.
- A position-specific MID model will try to learn a single pattern across fundamentally different player roles.

**ML implication:** Consider sub-classifying MID using goals/90 + assists/90 ratios (attacking vs defensive mid), or using a clustering approach. Alternatively, use per-90 stats rather than raw totals to partially mitigate role differences.

---

## Bias 8 — Manager Mode Contamination (2024-25)

**Severity: LOW**

FPL introduced a manager game mode in 2024-25. Manager cards are present in `fact_gw_player` with fundamentally different stats (no goals, no assists, no xG — instead mng_win, mng_draw, mng_loss).

### Manager rows in 2024-25

| Category | Rows |
|----------|:----:|
| Manager rows (mng_win IS NOT NULL) | 322 |
| Regular player rows | 27,283 |
| Total 2024-25 rows | 27,605 |

Manager cards have `value` as low as 5 (£0.5m) and zero for all player performance columns.

**ML implication:** Filter with `WHERE mng_win IS NULL` before any modelling. Already fixed in the ETL (mng_* = NULL for non-manager rows), so `WHERE mng_win IS NULL` cleanly isolates regular players.

---

## Bias 9 — Fixture Difficulty (Missing Feature)

**Severity: CRITICAL**

Opponent quality has a large, measurable effect on individual player points. However, there is no `opponent_rank` or fixture difficulty rating in the raw FPL dataset — it must be engineered.

### Average points per appearance vs top-6 vs rest (all seasons)

Top-6 defined as: Arsenal, Chelsea, Liverpool, Man City, Man Utd, Spurs

| Position | vs Top-6 | vs Others | Difference |
|----------|:--------:|:---------:|:----------:|
| GK | 0.858 | 1.043 | **+21.6%** |
| DEF | 0.893 | 1.327 | **+48.6%** |
| MID | 1.161 | 1.397 | **+20.3%** |
| FWD | 1.152 | 1.501 | **+30.3%** |

**Findings:**
- Defenders face a 48.6% points penalty when playing top-6 opponents — the largest positional effect in the dataset.
- The effect is consistent and large enough to cause systematic misprediction for any model that doesn't account for fixture difficulty.
- Since top-6 teams rotate in and out (Newcastle may qualify in some seasons, Spurs may not), this needs a dynamic ranking approach, not a static team list.

**ML implication:** This is the most important missing feature. Engineer `opponent_season_rank` (1–20 by final league position per season) or `opponent_goals_scored_season` as a proxy for attacking strength when predicting defensive points. This feature must be created before modelling begins.

---

## Bias 10 — Transfer Activity Leakage

**Severity: MEDIUM**

`transfers_in` and `transfers_out` in `fact_gw_player` record how many FPL managers bought or sold a player that gameweek. This is a reactive signal — managers respond to performance, injury news, and fixture swings that they observe before the deadline.

### Average transfers_in by points scored in that same GW

| Points scored | Avg transfers_in |
|:-------------:|:----------------:|
| 15+ pts | 246,513 |
| 10–14 pts | 123,150 |
| 5–9 pts | 53,714 |
| 0–4 pts | 7,040 |
| < 0 pts | 7,040 |

**35x spike** in transfers_in after 15+ point GWs vs low-scoring GWs.

**Findings:**
- Transfer activity strongly correlates with the same-GW score — because managers observe fixture result news, team announcements, and injury reports before the GW deadline.
- Using `transfers_in` from GW N to predict GW N performance would constitute lookahead bias.
- `transfers_in` from GW N-1 is a legitimate feature (it captures pre-GW N market sentiment).

**ML implication:** Never use same-GW `transfers_in` or `transfers_out` as a feature. If used at all, lag by exactly 1 GW. Treat it as a crowd-wisdom signal for the *next* GW, not the current one.

---

## Feature Engineering Required Before Modelling

Three features that must be created from the existing database before ML training can begin:

| Feature | Source | Addresses |
|---------|--------|-----------|
| `opponent_season_rank` | Final league table per season, joined via `opponent_team_sk` | Bias 9 (fixture difficulty) |
| `team_goals_conceded_season` | SUM(goals_conceded) per team per season from `fact_gw_player` | Bias 2 (team strength) |
| `era_id` | Derived from `season_id`: 1=pre-xG (seasons 1–6), 2=xG era (seasons 7–10) | Bias 4 (temporal drift) |

---

## Pre-Training Data Checklist

### Filtering
- [ ] Remove manager cards: `WHERE mng_win IS NULL`
- [ ] Remove sparse player-seasons: `WHERE gw_count >= 5` (or minutes >= 200)
- [ ] Decide era scope: all 10 seasons or xG era only (2022-23+)

### Feature engineering
- [ ] Create `opponent_season_rank` (1–20 per season)
- [ ] Create `team_goals_conceded_season` for defensive normalisation
- [ ] Create `era_id` flag
- [ ] Lag `transfers_in` and `transfers_out` by 1 GW

### Model design
- [ ] Train position-specific models (GK, DEF, MID, FWD) — do not train cross-position
- [ ] Include `was_home` and `opponent_season_rank` as mandatory features
- [ ] Use time-series cross-validation (expanding window, not random split)
- [ ] Validate separately by position, home/away, and minutes-played bucket

### Known limitations to document
- [ ] Model performance will degrade for rotation/fringe players (training data skews elite)
- [ ] Pre-2022-23 predictions will lack xG-based features
- [ ] Defensive stats remain partially confounded by team quality even after normalisation
- [ ] MID predictions will have higher variance than other positions due to role heterogeneity
