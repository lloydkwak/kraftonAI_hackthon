"""Relative scoring, in one implementation.

Two things use this and they must not drift apart: the leaderboard a participant reads
during the contest and the final score. One code path produces the numbers participants
see and the numbers that rank them, so this ships even though a single submission has no
score of its own.

**The procedure.** Per *episode*, not after averaging: each item's value is divided by
the best value among the submissions that **completed that episode**, and those
per-episode ratios are then averaged over the condition's episodes. A submission that did
not complete an episode scores 0 on all three items for it (a safe timeout excepted, see
below) and does not enter that episode's best. If nobody completes an episode, everybody
scores 0 on it -- which costs nothing in the ranking, because a uniform factor cancels.

**Partial credit for a safe timeout.** A run that
reaches the time limit without touching anything -- outcome `timeout` -- scores `(k/N)**2` times the **lowest**
weighted score among the episode's completers, `k` the waypoints it reached in order and `N` the course's
waypoints. It never enters the items' bests, and its own fuel, clearance and time are not used: a hull that
stops early burns no fuel and keeps its distance, so its own ratios would read ~1 for doing less. The square
keeps the easy first waypoints cheap (a run that reaches two of six and parks earns 1/9 of the worst
completion); the factor keeps every timeout below every completion; a collision, out-of-bounds or submission
fault still scores 0; and **if nobody completes the episode, nobody gets partial credit**. The weights are
needed for "weighted score", so partial credit applies only when `episode_ratios` is given them (the scorer
always passes them; called without weights, it keeps the binary rule).

**The three items and their directions.** Fuel and time are *smaller is better*, so the
ratio is `best / mine`; clearance is *larger is better*, so it is `mine / best`. Each
ratio therefore lands in `(0, 1]` and 1 means "best in field on this episode".

**Two things deliberately absent.** There is no reference path and no reference time:
per-episode normalisation cancels course length, so no organiser-produced anchor
exists to divide by. And the item weights are **not** here -- they are
`contest.WEIGHTS`, and `condition_score` takes them as an argument with no default.
"""

from __future__ import annotations

import math

ITEMS = ("fuel", "clearance", "time")     # fuel = control effort

#: Which direction each item improves in. `-1` means smaller is better.
SENSE = {"fuel": -1, "clearance": +1, "time": -1}


#: The exponent on a timeout's progress fraction.
PARTIAL_EXPONENT = 2


def partial_credit(entry: dict, floor: float) -> float:
    """A timeout's score: `(k/N)**PARTIAL_EXPONENT * floor`, `floor` the lowest completer's weighted score."""
    n = int(entry.get("of") or 0)
    k = min(int(entry.get("waypoints") or 0), n)
    return 0.0 if n <= 0 else (k / n) ** PARTIAL_EXPONENT * floor


def episode_ratios(entries: dict, weights: dict | None = None) -> dict:
    """`{team: {outcome, fuel, clearance, time[, waypoints, of]}}` -> `{team: {item: ratio}}`.

    With `weights`, a `timeout` gets partial credit (module docstring): every one of its items is set to its
    episode score, so any weighted total of them is that score.

    `fuel` is the control effort (sum over ticks of the squared normalised change of the applied
    action; `sim.Result.fuel`), `time` is ticks (or seconds -- any monotone
    unit, since the ratio cancels it), `clearance` is the statistic `clearance_statistic` computes.
    """
    completers = {t: e for t, e in entries.items() if e.get("outcome") == "goal"}
    out = {t: {i: 0.0 for i in ITEMS} for t in entries}
    if not completers:
        return out                       # nobody completed: everybody scores 0
    for item in ITEMS:
        vals = [float(e[item]) for e in completers.values()]
        best = max(vals) if SENSE[item] > 0 else min(vals)
        for team, e in completers.items():
            v = float(e[item])
            if SENSE[item] > 0:
                out[team][item] = 0.0 if best <= 0.0 else min(1.0, v / best)
            else:
                out[team][item] = 0.0 if v <= 0.0 else min(1.0, best / v)
    if weights:
        total_w = sum(weights[i] for i in ITEMS) or 1.0
        floor = min(sum(weights[i] * out[t][i] for i in ITEMS) / total_w for t in completers)
        for team, e in entries.items():
            if e.get("outcome") == "timeout":
                s_ = partial_credit(e, floor)
                out[team] = {i: s_ for i in ITEMS}
    return out


def condition_score(per_episode: list, weights: dict) -> dict:
    """Average each submission's per-episode ratios over a condition, then weight the items.

    `per_episode` is a list of the dicts `episode_ratios` returns, one per episode. The
    average runs over **all** the condition's episodes, including the ones a submission
    failed, because a 0 there is the point of the completion term.
    """
    teams = sorted({t for ep in per_episode for t in ep})
    n = max(len(per_episode), 1)
    out = {}
    for team in teams:
        items = {i: sum(ep.get(team, {}).get(i, 0.0) for ep in per_episode) / n
                 for i in ITEMS}
        items["total"] = sum(weights[i] * items[i] for i in ITEMS)
        out[team] = items
    return out


def final_score(by_condition: dict, condition_weights: dict | None = None) -> dict:
    """A **weighted** mean of the condition scores, 1-1 and 1-2 at 20% each, 1-3 and 1-4
    at 30% each (`contest.CONDITION_WEIGHTS`).

    With no weights given, or for conditions the weights do not name, every condition
    counts once. The weights are renormalised over the conditions actually scored, so a
    board restricted to two conditions still lands in [0, 1] and keeps their ratio.

    A mean rather than a product, so abandoning one condition still leaves the other
    three's share, which per-condition publication makes visible.
    """
    teams = sorted({t for c in by_condition.values() for t in c})
    w = {c: float((condition_weights or {}).get(c, 1.0)) for c in by_condition}
    total = sum(w.values()) or 1.0
    return {t: sum(w[c] * by_condition[c].get(t, {}).get("total", 0.0) for c in by_condition) / total
            for t in teams}


def tie_break_key(team: str, total: float, waypoints: int, submitted_at: float):
    """The tie-break chain: full-precision score, then waypoints reached, then timestamp.

    Sorts descending on score, so the key negates. It orders the *bottom* of the table:
    scoring zeroes every item on a non-completion other than a partially credited timeout,
    so every submission with no completion and no partial credit has exactly the same
    score, and nothing else in a mean of continuous ratios ties.
    """
    return (-total, -waypoints, submitted_at, team)


def clearance_statistic(per_tick, percentile: float, d_cap: float) -> float:
    """The clearance statistic, with both parameters (`percentile`, `d_cap`) exposed.

    The cap is applied **per tick, before averaging**: capping the finished average
    instead mixes capped and uncapped ticks and leaves the value sensitive to how much
    open water the route happened to contain. Recomputable offline from
    `Result.per_tick_clearance`.
    """
    if not len(per_tick):
        return 0.0
    a = sorted(min(float(v), d_cap) for v in per_tick)
    k = max(1, int(math.ceil(len(a) * percentile / 100.0)))
    return sum(a[:k]) / k
