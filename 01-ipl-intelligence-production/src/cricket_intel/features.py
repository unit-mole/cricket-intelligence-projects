from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import math
import numpy as np
import pandas as pd

from .utils import dump_json

# Every feature is expressed from Team 1's perspective. Swapping Team 1/Team 2 should negate the vector.
PRETOSS_FEATURE_COLUMNS = [
    "elo_diff",
    "fast_elo_diff",
    "slow_elo_diff",
    "form5_diff",
    "form10_diff",
    "form20_diff",
    "form_ewm_diff",
    "season_form_diff",
    "opponent_adjusted_form_diff",
    "strength_of_schedule_diff",
    "venue_form_diff",
    "venue_experience_log_diff",
    "h2h_advantage",
    "runs10_diff",
    "run_rate10_diff",
    "score_consistency_advantage",
    "rr_consistency_advantage",
    "wickets_taken10_diff",
    "wickets_lost10_advantage",
    "batting_dot_pct_advantage",
    "batting_boundary_pct_diff",
    "bowling_dot_pct_diff",
    "bowling_boundary_pct_advantage",
    "pp_batting_rr_diff",
    "middle_batting_rr_diff",
    "death_batting_rr_diff",
    "pp_bowling_rr_advantage",
    "middle_bowling_rr_advantage",
    "death_bowling_rr_advantage",
    "pp_bowling_wicket_rate_diff",
    "middle_bowling_wicket_rate_diff",
    "death_bowling_wicket_rate_diff",
    "death_acceleration_diff",
    "chase_form_diff",
    "defend_form_diff",
    "experience_log_diff",
    "knockout_form_diff",
    "rest_days_diff",
    "squad_batting_runs_diff",
    "squad_batting_sr_diff",
    "squad_bowling_wickets_diff",
    "squad_bowling_economy_advantage",
    "squad_experience_log_diff",
    "squad_continuity_diff",
    "squad_rookie_load_advantage",
]

POSTTOSS_EXTRA_COLUMNS = [
    "toss_advantage",
    "team1_chasing",
    "venue_chase_interaction",
    "lineup_batting_runs_diff",
    "lineup_batting_sr_diff",
    "lineup_bowling_wickets_diff",
    "lineup_bowling_economy_advantage",
    "lineup_experience_log_diff",
    "lineup_continuity_diff",
    "lineup_rookie_load_advantage",
]

POSTTOSS_FEATURE_COLUMNS = PRETOSS_FEATURE_COLUMNS + POSTTOSS_EXTRA_COLUMNS
FEATURE_COLUMNS = PRETOSS_FEATURE_COLUMNS  # backwards-friendly alias; tournament simulation uses pre-toss.


def feature_columns(mode: str) -> list[str]:
    mode = str(mode).lower().strip()
    if mode in {"pretoss", "pre", "pre_toss"}:
        return PRETOSS_FEATURE_COLUMNS
    if mode in {"posttoss", "post", "post_toss"}:
        return POSTTOSS_FEATURE_COLUMNS
    raise ValueError(f"Unknown mode: {mode}")


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float(default)


def _std(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return float(np.std(vals, ddof=0)) if len(vals) >= 2 else float(default)


def _ewm(values: Iterable[float], halflife: float = 4.0, default: float = 0.5) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return float(default)
    ages = np.arange(vals.size - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / max(halflife, 1e-6))
    return float(np.average(vals, weights=weights))


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den else float(default)


def _is_knockout(stage: str, event: str) -> bool:
    s = (str(stage) + " " + str(event)).lower()
    return any(k in s for k in ["final", "semi", "qualifier", "eliminator", "playoff", "knockout"])


def _parse_players(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split("|") if x.strip()]


@dataclass
class PlayerState:
    matches: int = 0
    last_team: str = ""
    last_date: pd.Timestamp | None = None
    team_matches: dict[str, int] = field(default_factory=dict)
    runs: deque = field(default_factory=lambda: deque(maxlen=12))
    balls: deque = field(default_factory=lambda: deque(maxlen=12))
    wickets: deque = field(default_factory=lambda: deque(maxlen=12))
    runs_conceded: deque = field(default_factory=lambda: deque(maxlen=12))
    balls_bowled: deque = field(default_factory=lambda: deque(maxlen=12))

    def update(self, row) -> None:
        team = str(row.team)
        self.matches += 1
        self.last_team = team
        self.last_date = pd.Timestamp(row.date)
        self.team_matches[team] = int(self.team_matches.get(team, 0)) + 1
        self.runs.append(float(getattr(row, "runs", 0.0) or 0.0))
        self.balls.append(float(getattr(row, "balls", 0.0) or 0.0))
        self.wickets.append(float(getattr(row, "wickets", 0.0) or 0.0))
        self.runs_conceded.append(float(getattr(row, "runs_conceded", 0.0) or 0.0))
        self.balls_bowled.append(float(getattr(row, "balls_bowled", 0.0) or 0.0))

    def metrics(self, team: str | None = None) -> dict[str, float]:
        runs = float(np.sum(self.runs))
        balls = float(np.sum(self.balls))
        wkts = float(np.sum(self.wickets))
        conceded = float(np.sum(self.runs_conceded))
        bowl_balls = float(np.sum(self.balls_bowled))
        n = max(len(self.runs), 1)
        continuity = _safe_ratio(self.team_matches.get(team or self.last_team, 0), self.matches, 0.0)
        return {
            "runs_per_match": runs / n,
            "strike_rate": 100.0 * runs / balls if balls else 115.0,
            "wickets_per_match": wkts / n,
            "economy": 6.0 * conceded / bowl_balls if bowl_balls else 8.5,
            "matches": float(self.matches),
            "continuity": continuity,
            "rookie": 1.0 if self.matches < 10 else 0.0,
        }


@dataclass
class TeamState:
    elo: float = 1500.0
    fast_elo: float = 1500.0
    slow_elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=30))
    season_results: list = field(default_factory=list)
    opponent_adjusted: deque = field(default_factory=lambda: deque(maxlen=20))
    opponent_elo: deque = field(default_factory=lambda: deque(maxlen=20))
    scores: deque = field(default_factory=lambda: deque(maxlen=15))
    run_rates: deque = field(default_factory=lambda: deque(maxlen=15))
    wickets_taken: deque = field(default_factory=lambda: deque(maxlen=15))
    wickets_lost: deque = field(default_factory=lambda: deque(maxlen=15))
    bat_dot_pct: deque = field(default_factory=lambda: deque(maxlen=15))
    bat_boundary_pct: deque = field(default_factory=lambda: deque(maxlen=15))
    bowl_dot_pct: deque = field(default_factory=lambda: deque(maxlen=15))
    bowl_boundary_pct: deque = field(default_factory=lambda: deque(maxlen=15))
    pp_bat_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    middle_bat_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    death_bat_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    pp_bowl_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    middle_bowl_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    death_bowl_rr: deque = field(default_factory=lambda: deque(maxlen=15))
    pp_bowl_wicket_rate: deque = field(default_factory=lambda: deque(maxlen=15))
    middle_bowl_wicket_rate: deque = field(default_factory=lambda: deque(maxlen=15))
    death_bowl_wicket_rate: deque = field(default_factory=lambda: deque(maxlen=15))
    chase: deque = field(default_factory=lambda: deque(maxlen=20))
    defend: deque = field(default_factory=lambda: deque(maxlen=20))
    knockout: deque = field(default_factory=lambda: deque(maxlen=20))
    last_date: pd.Timestamp | None = None


class PointInTimeFeatureBuilderV2:
    def __init__(self, season_reversion: float = 0.78, player_recent_days: int = 550):
        self.season_reversion = float(season_reversion)
        self.player_recent_days = int(player_recent_days)
        self.teams: defaultdict[str, TeamState] = defaultdict(TeamState)
        self.players: defaultdict[str, PlayerState] = defaultdict(PlayerState)
        self.venue_team: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self.venue_global: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {"matches": 0.0, "chase_wins": 0.0, "first_runs_sum": 0.0, "first_runs_n": 0.0}
        )
        self.h2h: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
        self.current_season: str | None = None
        self.as_of_date: pd.Timestamp | None = None

    def prepare_season(self, season: str) -> None:
        season = str(season)
        if self.current_season == season:
            return
        if self.current_season is not None:
            for state in self.teams.values():
                state.elo = 1500.0 + self.season_reversion * (state.elo - 1500.0)
                state.fast_elo = 1500.0 + self.season_reversion * (state.fast_elo - 1500.0)
                state.slow_elo = 1500.0 + self.season_reversion * (state.slow_elo - 1500.0)
                state.season_results = []
        self.current_season = season

    def _form(self, team: str, n: int) -> float:
        return _mean(list(self.teams[team].recent)[-n:], 0.5)

    def _venue_rate(self, team: str, venue: str) -> float:
        wins, matches = self.venue_team[(team, venue)]
        return float((wins + 2.0) / (matches + 4.0))

    def _venue_experience(self, team: str, venue: str) -> float:
        return float(self.venue_team[(team, venue)][1])

    def _venue_chase_rate(self, venue: str) -> float:
        d = self.venue_global[venue]
        return float((d["chase_wins"] + 3.0) / (d["matches"] + 6.0))

    def _h2h_advantage(self, a: str, b: str) -> float:
        key = tuple(sorted([a, b]))
        wa, wb, matches = self.h2h[key]
        if matches == 0:
            return 0.0
        wins_a = wa if a == key[0] else wb
        return float(2.0 * ((wins_a + 1.5) / (matches + 3.0)) - 1.0)

    def _team_player_candidates(self, team: str, date: pd.Timestamp) -> list[str]:
        recent = []
        older = []
        for player, state in self.players.items():
            if state.last_team != team or state.last_date is None:
                continue
            age = (pd.Timestamp(date) - state.last_date).days
            item = (age, -state.matches, player)
            if age <= self.player_recent_days:
                recent.append(item)
            else:
                older.append(item)
        ordered = sorted(recent) + sorted(older)
        return [x[2] for x in ordered[:18]]

    def _aggregate_players(self, names: list[str], team: str) -> dict[str, float]:
        # Unknown/new players get conservative priors rather than zero, so rookie-heavy lineups are not absurdly penalized.
        metrics = []
        seen = set()
        for name in names:
            name = str(name).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            if name in self.players and self.players[name].matches > 0:
                metrics.append(self.players[name].metrics(team))
            else:
                metrics.append({
                    "runs_per_match": 12.0,
                    "strike_rate": 118.0,
                    "wickets_per_match": 0.25,
                    "economy": 8.7,
                    "matches": 0.0,
                    "continuity": 0.0,
                    "rookie": 1.0,
                })
        if not metrics:
            return {
                "batting_runs": 105.0,
                "batting_sr": 120.0,
                "bowling_wickets": 3.0,
                "bowling_economy": 8.5,
                "experience": 0.0,
                "continuity": 0.0,
                "rookie_load": 1.0,
                "players": 0.0,
            }

        bat = sorted(metrics, key=lambda m: (m["runs_per_match"], m["strike_rate"]), reverse=True)[:7]
        bowl = sorted(metrics, key=lambda m: (m["wickets_per_match"], -m["economy"]), reverse=True)[:5]
        top11 = sorted(metrics, key=lambda m: m["matches"], reverse=True)[:11]
        bat_runs = float(sum(m["runs_per_match"] for m in bat))
        bat_weight = sum(max(m["runs_per_match"], 3.0) for m in bat)
        bat_sr = float(sum(m["strike_rate"] * max(m["runs_per_match"], 3.0) for m in bat) / bat_weight) if bat_weight else 120.0
        bowl_wickets = float(sum(m["wickets_per_match"] for m in bowl))
        bowl_weight = sum(max(m["wickets_per_match"], 0.15) for m in bowl)
        bowl_economy = float(sum(m["economy"] * max(m["wickets_per_match"], 0.15) for m in bowl) / bowl_weight) if bowl_weight else 8.5
        return {
            "batting_runs": bat_runs,
            "batting_sr": bat_sr,
            "bowling_wickets": bowl_wickets,
            "bowling_economy": bowl_economy,
            "experience": float(sum(m["matches"] for m in top11)),
            "continuity": _mean([m["continuity"] for m in top11], 0.0),
            "rookie_load": _mean([m["rookie"] for m in top11], 1.0),
            "players": float(len(metrics)),
        }

    def _squad_metrics(self, team: str, date: pd.Timestamp) -> dict[str, float]:
        return self._aggregate_players(self._team_player_candidates(team, date), team)

    def _posttoss_context(self, a: str, b: str, venue: str, toss_winner: str, toss_decision: str) -> dict[str, float]:
        toss_winner = str(toss_winner or "")
        toss_decision = str(toss_decision or "").lower()
        toss_adv = 1.0 if toss_winner == a else (-1.0 if toss_winner == b else 0.0)
        team1_chasing = 0.0
        if toss_winner in {a, b} and toss_decision in {"bat", "field"}:
            if toss_winner == a:
                team1_chasing = 1.0 if toss_decision == "field" else -1.0
            else:
                team1_chasing = 1.0 if toss_decision == "bat" else -1.0
        chase_centered = 2.0 * (self._venue_chase_rate(venue) - 0.5)
        return {
            "toss_advantage": toss_adv,
            "team1_chasing": team1_chasing,
            "venue_chase_interaction": team1_chasing * chase_centered,
        }

    def features(
        self,
        a: str,
        b: str,
        venue: str,
        date: pd.Timestamp,
        toss_winner: str = "",
        toss_decision: str = "",
        lineup_a: list[str] | None = None,
        lineup_b: list[str] | None = None,
    ) -> dict[str, float]:
        A, B = self.teams[a], self.teams[b]
        date = pd.Timestamp(date)
        rest_a = (date - A.last_date).days if A.last_date is not None else 14
        rest_b = (date - B.last_date).days if B.last_date is not None else 14

        squad_a = self._squad_metrics(a, date)
        squad_b = self._squad_metrics(b, date)
        lineup_a_metrics = self._aggregate_players(lineup_a or self._team_player_candidates(a, date), a)
        lineup_b_metrics = self._aggregate_players(lineup_b or self._team_player_candidates(b, date), b)

        feat = {
            "elo_diff": A.elo - B.elo,
            "fast_elo_diff": A.fast_elo - B.fast_elo,
            "slow_elo_diff": A.slow_elo - B.slow_elo,
            "form5_diff": self._form(a, 5) - self._form(b, 5),
            "form10_diff": self._form(a, 10) - self._form(b, 10),
            "form20_diff": self._form(a, 20) - self._form(b, 20),
            "form_ewm_diff": _ewm(A.recent, 4.0, 0.5) - _ewm(B.recent, 4.0, 0.5),
            "season_form_diff": _mean(A.season_results, 0.5) - _mean(B.season_results, 0.5),
            "opponent_adjusted_form_diff": _ewm(A.opponent_adjusted, 5.0, 0.0) - _ewm(B.opponent_adjusted, 5.0, 0.0),
            "strength_of_schedule_diff": (_mean(A.opponent_elo, 1500.0) - _mean(B.opponent_elo, 1500.0)) / 100.0,
            "venue_form_diff": self._venue_rate(a, venue) - self._venue_rate(b, venue),
            "venue_experience_log_diff": math.log1p(self._venue_experience(a, venue)) - math.log1p(self._venue_experience(b, venue)),
            "h2h_advantage": self._h2h_advantage(a, b),
            "runs10_diff": _mean(list(A.scores)[-10:], 160.0) - _mean(list(B.scores)[-10:], 160.0),
            "run_rate10_diff": _mean(list(A.run_rates)[-10:], 8.0) - _mean(list(B.run_rates)[-10:], 8.0),
            "score_consistency_advantage": _std(list(B.scores)[-10:], 25.0) - _std(list(A.scores)[-10:], 25.0),
            "rr_consistency_advantage": _std(list(B.run_rates)[-10:], 1.2) - _std(list(A.run_rates)[-10:], 1.2),
            "wickets_taken10_diff": _mean(list(A.wickets_taken)[-10:], 6.0) - _mean(list(B.wickets_taken)[-10:], 6.0),
            "wickets_lost10_advantage": _mean(list(B.wickets_lost)[-10:], 6.0) - _mean(list(A.wickets_lost)[-10:], 6.0),
            "batting_dot_pct_advantage": _mean(A.bat_dot_pct, 0.35) * -1.0 + _mean(B.bat_dot_pct, 0.35),
            "batting_boundary_pct_diff": _mean(A.bat_boundary_pct, 0.10) - _mean(B.bat_boundary_pct, 0.10),
            "bowling_dot_pct_diff": _mean(A.bowl_dot_pct, 0.35) - _mean(B.bowl_dot_pct, 0.35),
            "bowling_boundary_pct_advantage": _mean(B.bowl_boundary_pct, 0.10) - _mean(A.bowl_boundary_pct, 0.10),
            "pp_batting_rr_diff": _mean(A.pp_bat_rr, 8.0) - _mean(B.pp_bat_rr, 8.0),
            "middle_batting_rr_diff": _mean(A.middle_bat_rr, 8.0) - _mean(B.middle_bat_rr, 8.0),
            "death_batting_rr_diff": _mean(A.death_bat_rr, 9.5) - _mean(B.death_bat_rr, 9.5),
            "pp_bowling_rr_advantage": _mean(B.pp_bowl_rr, 8.0) - _mean(A.pp_bowl_rr, 8.0),
            "middle_bowling_rr_advantage": _mean(B.middle_bowl_rr, 8.0) - _mean(A.middle_bowl_rr, 8.0),
            "death_bowling_rr_advantage": _mean(B.death_bowl_rr, 9.5) - _mean(A.death_bowl_rr, 9.5),
            "pp_bowling_wicket_rate_diff": _mean(A.pp_bowl_wicket_rate, 0.35) - _mean(B.pp_bowl_wicket_rate, 0.35),
            "middle_bowling_wicket_rate_diff": _mean(A.middle_bowl_wicket_rate, 0.40) - _mean(B.middle_bowl_wicket_rate, 0.40),
            "death_bowling_wicket_rate_diff": _mean(A.death_bowl_wicket_rate, 0.65) - _mean(B.death_bowl_wicket_rate, 0.65),
            "death_acceleration_diff": (_mean(A.death_bat_rr, 9.5) - _mean(A.middle_bat_rr, 8.0)) - (_mean(B.death_bat_rr, 9.5) - _mean(B.middle_bat_rr, 8.0)),
            "chase_form_diff": _mean(A.chase, 0.5) - _mean(B.chase, 0.5),
            "defend_form_diff": _mean(A.defend, 0.5) - _mean(B.defend, 0.5),
            "experience_log_diff": math.log1p(A.matches) - math.log1p(B.matches),
            "knockout_form_diff": _mean(A.knockout, 0.5) - _mean(B.knockout, 0.5),
            "rest_days_diff": float(np.clip(rest_a - rest_b, -60, 60)),
            "squad_batting_runs_diff": squad_a["batting_runs"] - squad_b["batting_runs"],
            "squad_batting_sr_diff": squad_a["batting_sr"] - squad_b["batting_sr"],
            "squad_bowling_wickets_diff": squad_a["bowling_wickets"] - squad_b["bowling_wickets"],
            "squad_bowling_economy_advantage": squad_b["bowling_economy"] - squad_a["bowling_economy"],
            "squad_experience_log_diff": math.log1p(squad_a["experience"]) - math.log1p(squad_b["experience"]),
            "squad_continuity_diff": squad_a["continuity"] - squad_b["continuity"],
            "squad_rookie_load_advantage": squad_b["rookie_load"] - squad_a["rookie_load"],
            **self._posttoss_context(a, b, venue, toss_winner, toss_decision),
            "lineup_batting_runs_diff": lineup_a_metrics["batting_runs"] - lineup_b_metrics["batting_runs"],
            "lineup_batting_sr_diff": lineup_a_metrics["batting_sr"] - lineup_b_metrics["batting_sr"],
            "lineup_bowling_wickets_diff": lineup_a_metrics["bowling_wickets"] - lineup_b_metrics["bowling_wickets"],
            "lineup_bowling_economy_advantage": lineup_b_metrics["bowling_economy"] - lineup_a_metrics["bowling_economy"],
            "lineup_experience_log_diff": math.log1p(lineup_a_metrics["experience"]) - math.log1p(lineup_b_metrics["experience"]),
            "lineup_continuity_diff": lineup_a_metrics["continuity"] - lineup_b_metrics["continuity"],
            "lineup_rookie_load_advantage": lineup_b_metrics["rookie_load"] - lineup_a_metrics["rookie_load"],
        }
        return {k: float(feat[k]) for k in POSTTOSS_FEATURE_COLUMNS}

    def _update_elo(self, a: str, b: str, ya: int) -> tuple[float, float, float, float]:
        A, B = self.teams[a], self.teams[b]
        pre_a, pre_b = A.elo, B.elo
        expected = 1.0 / (1.0 + 10.0 ** ((B.elo - A.elo) / 400.0))
        for attr, k in [("elo", 24.0), ("fast_elo", 40.0), ("slow_elo", 12.0)]:
            ra, rb = getattr(A, attr), getattr(B, attr)
            exp = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            delta = k * (ya - exp)
            setattr(A, attr, ra + delta)
            setattr(B, attr, rb - delta)
        return expected, 1.0 - expected, pre_a, pre_b

    def update_match(self, row) -> None:
        a, b, winner = str(row.team1), str(row.team2), str(row.winner)
        date = pd.Timestamp(row.date)
        ya = 1 if winner == a else 0
        yb = 1 - ya
        A, B = self.teams[a], self.teams[b]
        exp_a, exp_b, pre_a, pre_b = self._update_elo(a, b, ya)

        for team, state, outcome, opp_elo, exp in [
            (a, A, ya, pre_b, exp_a),
            (b, B, yb, pre_a, exp_b),
        ]:
            state.matches += 1
            state.wins += outcome
            state.recent.append(outcome)
            state.season_results.append(outcome)
            state.opponent_adjusted.append(float(outcome - exp))
            state.opponent_elo.append(float(opp_elo))
            state.last_date = date
            vw, vm = self.venue_team[(team, str(row.venue))]
            self.venue_team[(team, str(row.venue))] = [vw + outcome, vm + 1]
            if _is_knockout(getattr(row, "stage", ""), getattr(row, "event_name", "")):
                state.knockout.append(outcome)

        key = tuple(sorted([a, b]))
        x, y, m = self.h2h[key]
        winning_index = 0 if winner == key[0] else 1
        if winning_index == 0:
            x += 1
        else:
            y += 1
        self.h2h[key] = [x, y, m + 1]

        # Rich innings/phase statistics are post-match updates only; they can influence later matches, never the current row.
        def append_batting(state: TeamState, prefix: str) -> None:
            runs = float(getattr(row, f"{prefix}_runs", 0.0) or 0.0)
            overs = float(getattr(row, f"{prefix}_overs", 0.0) or 0.0)
            if overs <= 0:
                return
            state.scores.append(runs)
            state.run_rates.append(float(getattr(row, f"{prefix}_run_rate", runs / overs) or 0.0))
            state.wickets_lost.append(float(getattr(row, f"{prefix}_wickets", 0.0) or 0.0))
            state.bat_dot_pct.append(float(getattr(row, f"{prefix}_dot_pct", 0.35) or 0.0))
            state.bat_boundary_pct.append(float(getattr(row, f"{prefix}_boundary_pct", 0.10) or 0.0))
            state.pp_bat_rr.append(float(getattr(row, f"{prefix}_pp_rr", 8.0) or 0.0))
            state.middle_bat_rr.append(float(getattr(row, f"{prefix}_middle_rr", 8.0) or 0.0))
            state.death_bat_rr.append(float(getattr(row, f"{prefix}_death_rr", 9.5) or 0.0))

        append_batting(A, "team1")
        append_batting(B, "team2")

        # A's bowling equals B's batting innings and vice versa.
        if float(getattr(row, "team2_overs", 0.0) or 0.0) > 0:
            A.wickets_taken.append(float(getattr(row, "team2_wickets", 0.0) or 0.0))
            A.bowl_dot_pct.append(float(getattr(row, "team2_dot_pct", 0.35) or 0.0))
            A.bowl_boundary_pct.append(float(getattr(row, "team2_boundary_pct", 0.10) or 0.0))
            A.pp_bowl_rr.append(float(getattr(row, "team2_pp_rr", 8.0) or 0.0))
            A.middle_bowl_rr.append(float(getattr(row, "team2_middle_rr", 8.0) or 0.0))
            A.death_bowl_rr.append(float(getattr(row, "team2_death_rr", 9.5) or 0.0))
            A.pp_bowl_wicket_rate.append(float(getattr(row, "team2_pp_wicket_rate", 0.35) or 0.0))
            A.middle_bowl_wicket_rate.append(float(getattr(row, "team2_middle_wicket_rate", 0.40) or 0.0))
            A.death_bowl_wicket_rate.append(float(getattr(row, "team2_death_wicket_rate", 0.65) or 0.0))
        if float(getattr(row, "team1_overs", 0.0) or 0.0) > 0:
            B.wickets_taken.append(float(getattr(row, "team1_wickets", 0.0) or 0.0))
            B.bowl_dot_pct.append(float(getattr(row, "team1_dot_pct", 0.35) or 0.0))
            B.bowl_boundary_pct.append(float(getattr(row, "team1_boundary_pct", 0.10) or 0.0))
            B.pp_bowl_rr.append(float(getattr(row, "team1_pp_rr", 8.0) or 0.0))
            B.middle_bowl_rr.append(float(getattr(row, "team1_middle_rr", 8.0) or 0.0))
            B.death_bowl_rr.append(float(getattr(row, "team1_death_rr", 9.5) or 0.0))
            B.pp_bowl_wicket_rate.append(float(getattr(row, "team1_pp_wicket_rate", 0.35) or 0.0))
            B.middle_bowl_wicket_rate.append(float(getattr(row, "team1_middle_wicket_rate", 0.40) or 0.0))
            B.death_bowl_wicket_rate.append(float(getattr(row, "team1_death_wicket_rate", 0.65) or 0.0))

        first = str(getattr(row, "first_innings_team", "") or "")
        second = str(getattr(row, "second_innings_team", "") or "")
        if first in {a, b} and second in {a, b}:
            self.teams[second].chase.append(1 if winner == second else 0)
            self.teams[first].defend.append(1 if winner == first else 0)
            venue = self.venue_global[str(row.venue)]
            venue["matches"] += 1.0
            venue["chase_wins"] += 1.0 if winner == second else 0.0
            prefix = "team1" if first == a else "team2"
            fr = float(getattr(row, f"{prefix}_runs", 0.0) or 0.0)
            if fr > 0:
                venue["first_runs_sum"] += fr
                venue["first_runs_n"] += 1.0

        self.as_of_date = max(date, self.as_of_date) if self.as_of_date is not None else date

    def update_players(self, player_rows: pd.DataFrame) -> None:
        if player_rows is None or player_rows.empty:
            return
        for row in player_rows.itertuples(index=False):
            player = str(row.player).strip()
            if player:
                self.players[player].update(row)

    def _team_summary(self, team: str) -> dict[str, float | int | str | None]:
        s = self.teams[team]
        date = self.as_of_date or pd.Timestamp.today().normalize()
        squad = self._squad_metrics(team, date)
        return {
            "elo": s.elo,
            "fast_elo": s.fast_elo,
            "slow_elo": s.slow_elo,
            "matches": s.matches,
            "wins": s.wins,
            "form5": self._form(team, 5),
            "form10": self._form(team, 10),
            "form20": self._form(team, 20),
            "form_ewm": _ewm(s.recent, 4.0, 0.5),
            "season_form": _mean(s.season_results, 0.5),
            "opponent_adjusted_form": _ewm(s.opponent_adjusted, 5.0, 0.0),
            "strength_of_schedule": _mean(s.opponent_elo, 1500.0),
            "runs10": _mean(list(s.scores)[-10:], 160.0),
            "run_rate10": _mean(list(s.run_rates)[-10:], 8.0),
            "score_std10": _std(list(s.scores)[-10:], 25.0),
            "rr_std10": _std(list(s.run_rates)[-10:], 1.2),
            "wickets_taken10": _mean(list(s.wickets_taken)[-10:], 6.0),
            "wickets_lost10": _mean(list(s.wickets_lost)[-10:], 6.0),
            "bat_dot_pct": _mean(s.bat_dot_pct, 0.35),
            "bat_boundary_pct": _mean(s.bat_boundary_pct, 0.10),
            "bowl_dot_pct": _mean(s.bowl_dot_pct, 0.35),
            "bowl_boundary_pct": _mean(s.bowl_boundary_pct, 0.10),
            "pp_bat_rr": _mean(s.pp_bat_rr, 8.0),
            "middle_bat_rr": _mean(s.middle_bat_rr, 8.0),
            "death_bat_rr": _mean(s.death_bat_rr, 9.5),
            "pp_bowl_rr": _mean(s.pp_bowl_rr, 8.0),
            "middle_bowl_rr": _mean(s.middle_bowl_rr, 8.0),
            "death_bowl_rr": _mean(s.death_bowl_rr, 9.5),
            "pp_bowl_wicket_rate": _mean(s.pp_bowl_wicket_rate, 0.35),
            "middle_bowl_wicket_rate": _mean(s.middle_bowl_wicket_rate, 0.40),
            "death_bowl_wicket_rate": _mean(s.death_bowl_wicket_rate, 0.65),
            "chase_form": _mean(s.chase, 0.5),
            "defend_form": _mean(s.defend, 0.5),
            "knockout_form": _mean(s.knockout, 0.5),
            "last_date": str(s.last_date.date()) if s.last_date is not None else None,
            **{f"squad_{k}": v for k, v in squad.items() if k != "players"},
        }

    def snapshot(self, current_squads: dict[str, list[str]] | None = None) -> dict:
        teams = {team: self._team_summary(team) for team in self.teams}
        players = {}
        for name, state in self.players.items():
            m = state.metrics(state.last_team)
            players[name] = {
                **m,
                "last_team": state.last_team,
                "last_date": str(state.last_date.date()) if state.last_date is not None else None,
                "team_matches": state.team_matches,
            }
        return {
            "version": "2.0",
            "as_of_date": str(self.as_of_date.date()) if self.as_of_date is not None else None,
            "current_season": self.current_season,
            "teams": teams,
            "active_teams": sorted([team for team, state in self.teams.items() if len(state.season_results) > 0]),
            "players": players,
            "current_squads": current_squads or {},
            "venue_team": {f"{team}|||{venue}": vals for (team, venue), vals in self.venue_team.items()},
            "venue_global": dict(self.venue_global),
            "h2h": {f"{a}|||{b}": vals for (a, b), vals in self.h2h.items()},
            "venues": sorted({venue for (_, venue) in self.venue_team.keys() if venue}),
        }


def _load_current_squads(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"team", "player"}.issubset(df.columns):
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    for row in df.itertuples(index=False):
        team = str(row.team).strip()
        player = str(row.player).strip()
        if team and player:
            out[team].append(player)
    return dict(out)


def build_features(
    matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    out_dir: Path,
    season_reversion: float = 0.78,
    player_recent_days: int = 550,
    current_squads_path: Path | None = None,
) -> pd.DataFrame:
    df = matches.copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    players = player_stats.copy() if player_stats is not None else pd.DataFrame()
    if not players.empty:
        players["date"] = pd.to_datetime(players.date)
        by_match = {str(mid): g.copy() for mid, g in players.groupby(players.match_id.astype(str))}
    else:
        by_match = {}

    builder = PointInTimeFeatureBuilderV2(season_reversion=season_reversion, player_recent_days=player_recent_days)
    rows = []
    for row in df.itertuples(index=False):
        if row.winner not in {row.team1, row.team2}:
            continue
        builder.prepare_season(str(row.season))
        lineup_a = _parse_players(getattr(row, "team1_players", ""))
        lineup_b = _parse_players(getattr(row, "team2_players", ""))
        feat = builder.features(
            str(row.team1), str(row.team2), str(row.venue), pd.Timestamp(row.date),
            toss_winner=str(getattr(row, "toss_winner", "") or ""),
            toss_decision=str(getattr(row, "toss_decision", "") or ""),
            lineup_a=lineup_a,
            lineup_b=lineup_b,
        )
        feat.update({
            "match_id": str(row.match_id),
            "date": pd.Timestamp(row.date),
            "year": int(pd.Timestamp(row.date).year),
            "season": str(row.season),
            "team1": str(row.team1),
            "team2": str(row.team2),
            "venue": str(row.venue),
            "event_name": str(getattr(row, "event_name", "") or ""),
            "stage": str(getattr(row, "stage", "") or ""),
            "toss_winner": str(getattr(row, "toss_winner", "") or ""),
            "toss_decision": str(getattr(row, "toss_decision", "") or ""),
            "lineup_available": int(bool(lineup_a) and bool(lineup_b)),
            "target": 1 if row.winner == row.team1 else 0,
        })
        rows.append(feat)
        builder.update_match(row)
        builder.update_players(by_match.get(str(row.match_id), pd.DataFrame()))

    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "features_v2.csv", index=False)
    # Alias kept so standard scripts/tools can still locate one obvious feature table.
    out.to_csv(out_dir / "features.csv", index=False)
    schema = {
        "version": "2.0",
        "pretoss_feature_columns": PRETOSS_FEATURE_COLUMNS,
        "posttoss_feature_columns": POSTTOSS_FEATURE_COLUMNS,
        "feature_count_pretoss": len(PRETOSS_FEATURE_COLUMNS),
        "feature_count_posttoss": len(POSTTOSS_FEATURE_COLUMNS),
        "posttoss_lineup_note": "Historical exact XI is used when Cricsheet provides it; live inference can accept announced XI or fall back to the latest squad proxy.",
    }
    dump_json(schema, out_dir / "feature_schema_v2.json")
    dump_json(schema, out_dir / "feature_schema.json")
    squads = _load_current_squads(current_squads_path)
    snapshot = builder.snapshot(squads)
    dump_json(snapshot, out_dir / "latest_team_snapshot_v2.json")
    dump_json(snapshot, out_dir / "latest_team_snapshot.json")
    return out


def _snapshot_player_metrics(snapshot: dict, names: list[str], team: str) -> dict[str, float]:
    all_players = snapshot.get("players", {}) or {}
    metrics = []
    seen = set()
    for name in names:
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        p = all_players.get(name)
        if p:
            tm = p.get("team_matches", {}) or {}
            matches = float(p.get("matches", 0.0) or 0.0)
            metrics.append({
                "runs_per_match": float(p.get("runs_per_match", 12.0)),
                "strike_rate": float(p.get("strike_rate", 118.0)),
                "wickets_per_match": float(p.get("wickets_per_match", 0.25)),
                "economy": float(p.get("economy", 8.7)),
                "matches": matches,
                "continuity": float(tm.get(team, 0.0)) / matches if matches else 0.0,
                "rookie": 1.0 if matches < 10 else 0.0,
            })
        else:
            metrics.append({"runs_per_match": 12.0, "strike_rate": 118.0, "wickets_per_match": 0.25, "economy": 8.7, "matches": 0.0, "continuity": 0.0, "rookie": 1.0})
    if not metrics:
        return {"batting_runs": 105.0, "batting_sr": 120.0, "bowling_wickets": 3.0, "bowling_economy": 8.5, "experience": 0.0, "continuity": 0.0, "rookie_load": 1.0}
    bat = sorted(metrics, key=lambda m: (m["runs_per_match"], m["strike_rate"]), reverse=True)[:7]
    bowl = sorted(metrics, key=lambda m: (m["wickets_per_match"], -m["economy"]), reverse=True)[:5]
    top11 = sorted(metrics, key=lambda m: m["matches"], reverse=True)[:11]
    bw = sum(max(m["runs_per_match"], 3.0) for m in bat)
    ww = sum(max(m["wickets_per_match"], 0.15) for m in bowl)
    return {
        "batting_runs": float(sum(m["runs_per_match"] for m in bat)),
        "batting_sr": float(sum(m["strike_rate"] * max(m["runs_per_match"], 3.0) for m in bat) / bw) if bw else 120.0,
        "bowling_wickets": float(sum(m["wickets_per_match"] for m in bowl)),
        "bowling_economy": float(sum(m["economy"] * max(m["wickets_per_match"], 0.15) for m in bowl) / ww) if ww else 8.5,
        "experience": float(sum(m["matches"] for m in top11)),
        "continuity": _mean([m["continuity"] for m in top11], 0.0),
        "rookie_load": _mean([m["rookie"] for m in top11], 1.0),
    }


def _snapshot_squad_names(snapshot: dict, team: str) -> list[str]:
    current = (snapshot.get("current_squads", {}) or {}).get(team, [])
    if current:
        return list(current)
    players = snapshot.get("players", {}) or {}
    items = []
    as_of = pd.Timestamp(snapshot.get("as_of_date")) if snapshot.get("as_of_date") else pd.Timestamp.today().normalize()
    for name, p in players.items():
        if p.get("last_team") != team or not p.get("last_date"):
            continue
        age = (as_of - pd.Timestamp(p["last_date"])).days
        items.append((age, -float(p.get("matches", 0.0)), name))
    return [x[2] for x in sorted(items)[:18]]


def matchup_from_snapshot(
    snapshot: dict,
    team1: str,
    team2: str,
    venue: str = "",
    mode: str = "pretoss",
    toss_winner: str = "",
    toss_decision: str = "",
    lineup1: list[str] | None = None,
    lineup2: list[str] | None = None,
) -> pd.DataFrame:
    A = snapshot["teams"][team1]
    B = snapshot["teams"][team2]
    venue = str(venue or "")
    vt = snapshot.get("venue_team", {}) or {}
    vg = snapshot.get("venue_global", {}) or {}
    h2h = snapshot.get("h2h", {}) or {}

    def vvals(team: str):
        wins, matches = vt.get(f"{team}|||{venue}", [0, 0])
        return float((wins + 2.0) / (matches + 4.0)), float(matches)

    va, vexpa = vvals(team1)
    vb, vexpb = vvals(team2)
    key = "|||".join(sorted([team1, team2]))
    hw1, hw2, hm = h2h.get(key, [0, 0, 0])
    if hm:
        wins_team1 = hw1 if team1 == sorted([team1, team2])[0] else hw2
        hadv = 2.0 * ((wins_team1 + 1.5) / (hm + 3.0)) - 1.0
    else:
        hadv = 0.0

    as_of = pd.Timestamp(snapshot.get("as_of_date")) if snapshot.get("as_of_date") else pd.Timestamp.today().normalize()
    last_a = pd.Timestamp(A["last_date"]) if A.get("last_date") else as_of - pd.Timedelta(days=14)
    last_b = pd.Timestamp(B["last_date"]) if B.get("last_date") else as_of - pd.Timedelta(days=14)
    squad_a = _snapshot_player_metrics(snapshot, _snapshot_squad_names(snapshot, team1), team1)
    squad_b = _snapshot_player_metrics(snapshot, _snapshot_squad_names(snapshot, team2), team2)
    line_a = _snapshot_player_metrics(snapshot, lineup1 or _snapshot_squad_names(snapshot, team1), team1)
    line_b = _snapshot_player_metrics(snapshot, lineup2 or _snapshot_squad_names(snapshot, team2), team2)

    f = {
        "elo_diff": A["elo"] - B["elo"],
        "fast_elo_diff": A["fast_elo"] - B["fast_elo"],
        "slow_elo_diff": A["slow_elo"] - B["slow_elo"],
        "form5_diff": A["form5"] - B["form5"],
        "form10_diff": A["form10"] - B["form10"],
        "form20_diff": A["form20"] - B["form20"],
        "form_ewm_diff": A["form_ewm"] - B["form_ewm"],
        "season_form_diff": A["season_form"] - B["season_form"],
        "opponent_adjusted_form_diff": A["opponent_adjusted_form"] - B["opponent_adjusted_form"],
        "strength_of_schedule_diff": (A["strength_of_schedule"] - B["strength_of_schedule"]) / 100.0,
        "venue_form_diff": va - vb,
        "venue_experience_log_diff": math.log1p(vexpa) - math.log1p(vexpb),
        "h2h_advantage": hadv,
        "runs10_diff": A["runs10"] - B["runs10"],
        "run_rate10_diff": A["run_rate10"] - B["run_rate10"],
        "score_consistency_advantage": B["score_std10"] - A["score_std10"],
        "rr_consistency_advantage": B["rr_std10"] - A["rr_std10"],
        "wickets_taken10_diff": A["wickets_taken10"] - B["wickets_taken10"],
        "wickets_lost10_advantage": B["wickets_lost10"] - A["wickets_lost10"],
        "batting_dot_pct_advantage": B["bat_dot_pct"] - A["bat_dot_pct"],
        "batting_boundary_pct_diff": A["bat_boundary_pct"] - B["bat_boundary_pct"],
        "bowling_dot_pct_diff": A["bowl_dot_pct"] - B["bowl_dot_pct"],
        "bowling_boundary_pct_advantage": B["bowl_boundary_pct"] - A["bowl_boundary_pct"],
        "pp_batting_rr_diff": A["pp_bat_rr"] - B["pp_bat_rr"],
        "middle_batting_rr_diff": A["middle_bat_rr"] - B["middle_bat_rr"],
        "death_batting_rr_diff": A["death_bat_rr"] - B["death_bat_rr"],
        "pp_bowling_rr_advantage": B["pp_bowl_rr"] - A["pp_bowl_rr"],
        "middle_bowling_rr_advantage": B["middle_bowl_rr"] - A["middle_bowl_rr"],
        "death_bowling_rr_advantage": B["death_bowl_rr"] - A["death_bowl_rr"],
        "pp_bowling_wicket_rate_diff": A["pp_bowl_wicket_rate"] - B["pp_bowl_wicket_rate"],
        "middle_bowling_wicket_rate_diff": A["middle_bowl_wicket_rate"] - B["middle_bowl_wicket_rate"],
        "death_bowling_wicket_rate_diff": A["death_bowl_wicket_rate"] - B["death_bowl_wicket_rate"],
        "death_acceleration_diff": (A["death_bat_rr"] - A["middle_bat_rr"]) - (B["death_bat_rr"] - B["middle_bat_rr"]),
        "chase_form_diff": A["chase_form"] - B["chase_form"],
        "defend_form_diff": A["defend_form"] - B["defend_form"],
        "experience_log_diff": math.log1p(A["matches"]) - math.log1p(B["matches"]),
        "knockout_form_diff": A["knockout_form"] - B["knockout_form"],
        "rest_days_diff": float(np.clip((as_of - last_a).days - (as_of - last_b).days, -60, 60)),
        "squad_batting_runs_diff": squad_a["batting_runs"] - squad_b["batting_runs"],
        "squad_batting_sr_diff": squad_a["batting_sr"] - squad_b["batting_sr"],
        "squad_bowling_wickets_diff": squad_a["bowling_wickets"] - squad_b["bowling_wickets"],
        "squad_bowling_economy_advantage": squad_b["bowling_economy"] - squad_a["bowling_economy"],
        "squad_experience_log_diff": math.log1p(squad_a["experience"]) - math.log1p(squad_b["experience"]),
        "squad_continuity_diff": squad_a["continuity"] - squad_b["continuity"],
        "squad_rookie_load_advantage": squad_b["rookie_load"] - squad_a["rookie_load"],
    }

    toss_adv = 1.0 if toss_winner == team1 else (-1.0 if toss_winner == team2 else 0.0)
    chasing = 0.0
    dec = str(toss_decision or "").lower()
    if toss_winner in {team1, team2} and dec in {"bat", "field"}:
        if toss_winner == team1:
            chasing = 1.0 if dec == "field" else -1.0
        else:
            chasing = 1.0 if dec == "bat" else -1.0
    venue_rec = vg.get(venue, {}) or {}
    chase_rate = (float(venue_rec.get("chase_wins", 0.0)) + 3.0) / (float(venue_rec.get("matches", 0.0)) + 6.0)
    f.update({
        "toss_advantage": toss_adv,
        "team1_chasing": chasing,
        "venue_chase_interaction": chasing * 2.0 * (chase_rate - 0.5),
        "lineup_batting_runs_diff": line_a["batting_runs"] - line_b["batting_runs"],
        "lineup_batting_sr_diff": line_a["batting_sr"] - line_b["batting_sr"],
        "lineup_bowling_wickets_diff": line_a["bowling_wickets"] - line_b["bowling_wickets"],
        "lineup_bowling_economy_advantage": line_b["bowling_economy"] - line_a["bowling_economy"],
        "lineup_experience_log_diff": math.log1p(line_a["experience"]) - math.log1p(line_b["experience"]),
        "lineup_continuity_diff": line_a["continuity"] - line_b["continuity"],
        "lineup_rookie_load_advantage": line_b["rookie_load"] - line_a["rookie_load"],
    })
    cols = feature_columns(mode)
    return pd.DataFrame([[float(f[c]) for c in cols]], columns=cols)
