from __future__ import annotations
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd

from .features import matchup_from_snapshot


def pwin(bundle, snapshot: dict, a: str, b: str, venue: str = "") -> float:
    if a == b:
        return 0.5
    X = matchup_from_snapshot(snapshot, a, b, venue=venue, mode="pretoss")
    return float(bundle.predict_proba(X)[:, 1][0])


def build_probability_cache(bundle, snapshot: dict, teams: list[str], venue: str = "") -> dict[tuple[str, str], float]:
    cache = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            p = pwin(bundle, snapshot, a, b, venue=venue)
            cache[(a, b)] = p
            cache[(b, a)] = 1.0 - p
    return cache


def play_cached(cache, a: str, b: str, rng: np.random.Generator) -> str:
    return a if rng.random() < cache[(a, b)] else b


def _standings(teams: list[str], games: list[tuple[str, str, str]], snapshot: dict) -> list[str]:
    points = defaultdict(int)
    wins = defaultdict(int)
    for a, b, winner in games:
        points[winner] += 2
        wins[winner] += 1
    # V2 still avoids pretending to simulate exact NRR. Elo is a deterministic fallback tie-break only.
    return sorted(teams, key=lambda t: (points[t], wins[t], snapshot["teams"][t]["elo"]), reverse=True)


def active_ipl_teams(snapshot: dict) -> list[str]:
    active = [t for t in snapshot.get("active_teams", []) if t in snapshot.get("teams", {})]
    if len(active) == 10:
        return sorted(active)
    dated = []
    for team, values in snapshot.get("teams", {}).items():
        if values.get("last_date"):
            dated.append((team, pd.Timestamp(values["last_date"])))
    if not dated:
        return sorted(snapshot.get("teams", {}))[:10]
    max_date = max(d for _, d in dated)
    recent = [t for t, d in dated if d >= max_date - pd.Timedelta(days=240)]
    if len(recent) >= 10:
        return sorted(recent, key=lambda t: snapshot["teams"][t]["elo"], reverse=True)[:10]
    return [t for t, _ in sorted(dated, key=lambda x: x[1], reverse=True)[:10]]


def simulate_ipl(bundle, snapshot: dict, teams: list[str] | None = None, n: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(teams or active_ipl_teams(snapshot))
    if len(teams) != 10:
        raise ValueError(f"IPL simulation expects 10 active teams; got {len(teams)}")

    # No future schedule/venue is fabricated. This is a neutral-venue, double-round-robin scenario until an official schedule is supplied.
    cache = build_probability_cache(bundle, snapshot, teams, venue="")
    champion = defaultdict(int)
    playoff = defaultdict(int)
    finalist = defaultdict(int)
    position = defaultdict(float)

    for _ in range(int(n)):
        league_games = []
        for a, b in itertools.combinations(teams, 2):
            league_games.append((a, b, play_cached(cache, a, b, rng)))
            league_games.append((a, b, play_cached(cache, a, b, rng)))
        order = _standings(teams, league_games, snapshot)
        for idx, team in enumerate(order):
            position[team] += idx + 1
        for team in order[:4]:
            playoff[team] += 1

        q1 = play_cached(cache, order[0], order[1], rng)
        q1_loser = order[1] if q1 == order[0] else order[0]
        eliminator = play_cached(cache, order[2], order[3], rng)
        q2 = play_cached(cache, q1_loser, eliminator, rng)
        finalist[q1] += 1
        finalist[q2] += 1
        champion[play_cached(cache, q1, q2, rng)] += 1

    out = pd.DataFrame([
        {
            "team": t,
            "playoff_probability": playoff[t] / n,
            "final_probability": finalist[t] / n,
            "championship_probability": champion[t] / n,
            "expected_finish": position[t] / n,
        }
        for t in teams
    ])
    return out.sort_values("championship_probability", ascending=False).reset_index(drop=True)
