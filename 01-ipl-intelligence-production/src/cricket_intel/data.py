from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Any
import json
import re
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd

from .utils import sha256_file, dump_json

RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

NON_BOWLER_EXTRAS = {"byes", "legbyes", "penalty"}
NON_TEAM_WICKET_KINDS = {"retired hurt", "obstructing the field", "retired out"}
NON_BOWLER_WICKET_KINDS = NON_TEAM_WICKET_KINDS | {"run out"}


def clean_team(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    x = re.sub(r"\s+", " ", str(x)).strip()
    return RENAME.get(x, x)


def clean_player(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def download_zip(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with requests.get(url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def extract_zip(zip_path: Path, dest: Path, force: bool = False) -> Path:
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.rglob("*.json")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    return dest


def _phase(index: int) -> str:
    # Use the sequence position rather than relying on whether a provider stores over numbers 0- or 1-based.
    if index < 6:
        return "pp"
    if index < 15:
        return "middle"
    return "death"


def _blank_phase() -> dict[str, int]:
    return {"runs": 0, "wickets": 0, "balls": 0, "dots": 0, "boundaries": 0}


def _innings_summary(innings: dict) -> dict:
    total_runs = 0
    wickets = 0
    legal_balls = 0
    dots = 0
    boundaries = 0
    fours = 0
    sixes = 0
    phases = {"pp": _blank_phase(), "middle": _blank_phase(), "death": _blank_phase()}

    batting = defaultdict(lambda: {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dismissals": 0})
    bowling = defaultdict(lambda: {"runs_conceded": 0, "balls_bowled": 0, "wickets": 0, "dots_bowled": 0})

    for over_idx, over in enumerate(innings.get("overs", []) or []):
        ph = _phase(over_idx)
        for delivery in over.get("deliveries", []) or []:
            rr = delivery.get("runs", {}) or {}
            total = int(rr.get("total", 0) or 0)
            batter_runs = int(rr.get("batter", 0) or 0)
            extras = delivery.get("extras", {}) or {}
            legal = not ("wides" in extras or "noballs" in extras)
            batter = clean_player(delivery.get("batter", ""))
            bowler = clean_player(delivery.get("bowler", ""))

            total_runs += total
            phases[ph]["runs"] += total
            batting[batter]["runs"] += batter_runs
            if batter_runs == 4:
                fours += 1
                boundaries += 1
                phases[ph]["boundaries"] += 1
                batting[batter]["fours"] += 1
            elif batter_runs == 6:
                sixes += 1
                boundaries += 1
                phases[ph]["boundaries"] += 1
                batting[batter]["sixes"] += 1

            conceded = total - sum(int(extras.get(k, 0) or 0) for k in NON_BOWLER_EXTRAS)
            bowling[bowler]["runs_conceded"] += max(0, conceded)

            if legal:
                legal_balls += 1
                phases[ph]["balls"] += 1
                batting[batter]["balls"] += 1
                bowling[bowler]["balls_bowled"] += 1
                if total == 0:
                    dots += 1
                    phases[ph]["dots"] += 1
                    bowling[bowler]["dots_bowled"] += 1

            for wicket in delivery.get("wickets", []) or []:
                kind = str(wicket.get("kind", "")).lower()
                player_out = clean_player(wicket.get("player_out", ""))
                if kind not in NON_TEAM_WICKET_KINDS:
                    wickets += 1
                    phases[ph]["wickets"] += 1
                    if player_out:
                        batting[player_out]["dismissals"] += 1
                if kind not in NON_BOWLER_WICKET_KINDS and bowler:
                    bowling[bowler]["wickets"] += 1

    def ratio(num: float, den: float, mult: float = 1.0) -> float:
        return float(mult * num / den) if den else 0.0

    phase_metrics: dict[str, float] = {}
    for ph, v in phases.items():
        phase_metrics[f"{ph}_runs"] = float(v["runs"])
        phase_metrics[f"{ph}_wickets"] = float(v["wickets"])
        phase_metrics[f"{ph}_balls"] = float(v["balls"])
        phase_metrics[f"{ph}_rr"] = ratio(v["runs"], v["balls"], 6.0)
        phase_metrics[f"{ph}_dot_pct"] = ratio(v["dots"], v["balls"])
        phase_metrics[f"{ph}_boundary_pct"] = ratio(v["boundaries"], v["balls"])
        phase_metrics[f"{ph}_wicket_rate"] = ratio(v["wickets"], v["balls"], 6.0)

    return {
        "runs": float(total_runs),
        "wickets": float(wickets),
        "legal_balls": float(legal_balls),
        "overs": float(legal_balls / 6.0) if legal_balls else 0.0,
        "run_rate": ratio(total_runs, legal_balls, 6.0),
        "dot_pct": ratio(dots, legal_balls),
        "boundary_pct": ratio(boundaries, legal_balls),
        "wicket_rate": ratio(wickets, legal_balls, 6.0),
        "fours": float(fours),
        "sixes": float(sixes),
        **phase_metrics,
        "batting": dict(batting),
        "bowling": dict(bowling),
    }


def _player_list(info: dict, team: str) -> list[str]:
    players = info.get("players", {}) or {}
    # Team strings in Cricsheet player dictionaries use the original team spelling.
    for original_name, values in players.items():
        if clean_team(original_name) == team:
            return [clean_player(p) for p in (values or []) if clean_player(p)]
    return []


def parse_cricsheet_json(folder: Path, project: str = "ipl") -> tuple[pd.DataFrame, pd.DataFrame]:
    matches: list[dict] = []
    player_rows: list[dict] = []

    files = sorted(folder.rglob("*.json"))
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        info = obj.get("info", {}) or {}
        gender = str(info.get("gender", "")).lower()
        teams = [clean_team(t) for t in info.get("teams", []) or []]
        if project == "world_cup" and gender not in {"male", "men"}:
            continue
        if len(teams) != 2 or not all(teams):
            continue

        dates = info.get("dates") or []
        date = str(dates[0]) if dates else ""
        outcome = info.get("outcome", {}) or {}
        winner = clean_team(outcome.get("winner", ""))
        if winner not in teams:
            # Ties/no-results are not used as binary target rows.
            continue

        event = info.get("event", {}) or {}
        event_name = str(event.get("name", ""))
        stage = str(event.get("stage", ""))
        venue = str(info.get("venue", "") or "").strip()
        city = str(info.get("city", "") or "").strip()
        toss = info.get("toss", {}) or {}
        toss_winner = clean_team(toss.get("winner", ""))
        toss_decision = str(toss.get("decision", "") or "").lower().strip()

        innings_summaries: dict[str, dict] = {}
        innings_order: list[str] = []
        for inn in (obj.get("innings", []) or [])[:2]:
            batting_team = clean_team(inn.get("team", ""))
            if batting_team not in teams:
                continue
            innings_order.append(batting_team)
            innings_summaries[batting_team] = _innings_summary(inn)

        row: dict[str, Any] = {
            "match_id": fp.stem,
            "date": date,
            "season": str(info.get("season", "")),
            "team1": teams[0],
            "team2": teams[1],
            "winner": winner,
            "venue": venue,
            "city": city,
            "event_name": event_name,
            "stage": stage,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "first_innings_team": innings_order[0] if innings_order else "",
            "second_innings_team": innings_order[1] if len(innings_order) > 1 else "",
        }

        player_match: dict[tuple[str, str], dict] = {}
        for team in teams:
            lineup = _player_list(info, team)
            row[f"{team == teams[0] and 'team1' or 'team2'}_players"] = "|".join(lineup)
            for player in lineup:
                player_match[(team, player)] = {
                    "date": date,
                    "season": str(info.get("season", "")),
                    "match_id": fp.stem,
                    "team": team,
                    "player": player,
                    "appeared": 1,
                    "runs": 0.0,
                    "balls": 0.0,
                    "fours": 0.0,
                    "sixes": 0.0,
                    "dismissals": 0.0,
                    "wickets": 0.0,
                    "runs_conceded": 0.0,
                    "balls_bowled": 0.0,
                    "dots_bowled": 0.0,
                }

        for i, team in enumerate(teams, 1):
            summary = innings_summaries.get(team, {})
            scalar_fields = [
                "runs", "wickets", "overs", "run_rate", "dot_pct", "boundary_pct", "wicket_rate",
                "pp_runs", "pp_wickets", "pp_balls", "pp_rr", "pp_dot_pct", "pp_boundary_pct", "pp_wicket_rate",
                "middle_runs", "middle_wickets", "middle_balls", "middle_rr", "middle_dot_pct", "middle_boundary_pct", "middle_wicket_rate",
                "death_runs", "death_wickets", "death_balls", "death_rr", "death_dot_pct", "death_boundary_pct", "death_wicket_rate",
            ]
            for field in scalar_fields:
                row[f"team{i}_{field}"] = float(summary.get(field, 0.0) or 0.0)

            opponent = teams[1] if team == teams[0] else teams[0]
            for player, stats in (summary.get("batting", {}) or {}).items():
                rec = player_match.setdefault((team, player), {
                    "date": date, "season": str(info.get("season", "")), "match_id": fp.stem,
                    "team": team, "player": player, "appeared": 1,
                    "runs": 0.0, "balls": 0.0, "fours": 0.0, "sixes": 0.0, "dismissals": 0.0,
                    "wickets": 0.0, "runs_conceded": 0.0, "balls_bowled": 0.0, "dots_bowled": 0.0,
                })
                for key in ["runs", "balls", "fours", "sixes", "dismissals"]:
                    rec[key] += float(stats.get(key, 0) or 0)

            # Bowlers in this innings belong to the opponent.
            for player, stats in (summary.get("bowling", {}) or {}).items():
                rec = player_match.setdefault((opponent, player), {
                    "date": date, "season": str(info.get("season", "")), "match_id": fp.stem,
                    "team": opponent, "player": player, "appeared": 1,
                    "runs": 0.0, "balls": 0.0, "fours": 0.0, "sixes": 0.0, "dismissals": 0.0,
                    "wickets": 0.0, "runs_conceded": 0.0, "balls_bowled": 0.0, "dots_bowled": 0.0,
                })
                for key in ["wickets", "runs_conceded", "balls_bowled", "dots_bowled"]:
                    rec[key] += float(stats.get(key, 0) or 0)

        player_rows.extend(player_match.values())
        matches.append(row)

    m = pd.DataFrame(matches)
    p = pd.DataFrame(player_rows)
    if not m.empty:
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        m = m.dropna(subset=["date"]).sort_values(["date", "match_id"]).drop_duplicates("match_id").reset_index(drop=True)
    if not p.empty:
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        p = p.dropna(subset=["date"]).sort_values(["date", "match_id", "team", "player"]).reset_index(drop=True)
    return m, p


def load_legacy_ipl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "match_id": df["id"].astype(str),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "season": df["season"].astype(str),
        "team1": df["team1"].map(clean_team),
        "team2": df["team2"].map(clean_team),
        "winner": df["winner"].map(clean_team),
        "venue": df["venue"].fillna(""),
        "city": df["city"].fillna(""),
        "event_name": "Indian Premier League",
        "stage": df.get("match_type", pd.Series([""] * len(df))).fillna(""),
        "toss_winner": df["toss_winner"].map(clean_team),
        "toss_decision": df["toss_decision"].fillna("").astype(str).str.lower(),
        "first_innings_team": "",
        "second_innings_team": "",
        "team1_players": "",
        "team2_players": "",
    })
    scalar_fields = [
        "runs", "wickets", "overs", "run_rate", "dot_pct", "boundary_pct", "wicket_rate",
        "pp_runs", "pp_wickets", "pp_balls", "pp_rr", "pp_dot_pct", "pp_boundary_pct", "pp_wicket_rate",
        "middle_runs", "middle_wickets", "middle_balls", "middle_rr", "middle_dot_pct", "middle_boundary_pct", "middle_wicket_rate",
        "death_runs", "death_wickets", "death_balls", "death_rr", "death_dot_pct", "death_boundary_pct", "death_wicket_rate",
    ]
    for i in (1, 2):
        for field in scalar_fields:
            out[f"team{i}_{field}"] = 0.0
    out = out.dropna(subset=["date"])
    out = out[(out.winner == out.team1) | (out.winner == out.team2)]
    return out.sort_values(["date", "match_id"]).reset_index(drop=True)


def validate_matches(df: pd.DataFrame, players: pd.DataFrame | None = None) -> dict:
    required = {"date", "team1", "team2", "winner", "venue", "match_id"}
    missing = sorted(required - set(df.columns))
    issues: list[str] = []
    if missing:
        issues.append(f"missing_columns={missing}")
    bad_same = bad_winner = dup = 0
    if not df.empty and not missing:
        bad_same = int((df.team1 == df.team2).sum())
        bad_winner = int(((df.winner != df.team1) & (df.winner != df.team2)).sum())
        dup = int(df.duplicated("match_id").sum())
        if bad_same:
            issues.append(f"same_team_rows={bad_same}")
        if bad_winner:
            issues.append(f"winner_not_in_team_universe={bad_winner}")
        if dup:
            issues.append(f"duplicate_match_ids={dup}")

    bbb_rows = int((pd.to_numeric(df.get("team1_overs", 0), errors="coerce").fillna(0) > 0).sum()) if len(df) else 0
    lineup_rows = 0
    if len(df) and "team1_players" in df and "team2_players" in df:
        lineup_rows = int(((df.team1_players.fillna("").astype(str).str.len() > 0) & (df.team2_players.fillna("").astype(str).str.len() > 0)).sum())

    return {
        "rows": int(len(df)),
        "date_min": str(df.date.min()) if len(df) else None,
        "date_max": str(df.date.max()) if len(df) else None,
        "teams": int(len(set(df.team1) | set(df.team2))) if len(df) else 0,
        "duplicate_match_ids": dup,
        "same_team_rows": bad_same,
        "winner_not_in_team_universe": bad_winner,
        "ball_by_ball_match_rows": bbb_rows,
        "ball_by_ball_coverage": float(bbb_rows / len(df)) if len(df) else 0.0,
        "lineup_match_rows": lineup_rows,
        "lineup_coverage": float(lineup_rows / len(df)) if len(df) else 0.0,
        "player_match_rows": int(len(players)) if players is not None else 0,
        "issues": issues,
        "status": "PASS" if not issues else "WARN",
    }


def build_data(project_root: Path, project: str, url: str, force: bool = False, allow_legacy_fallback: bool = True):
    ext = project_root / "data/external"
    raw = project_root / "data/raw/cricsheet_json"
    proc = project_root / "data/processed"
    meta = project_root / "data/metadata"
    proc.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    zip_path = ext / "cricsheet_ipl_json.zip"
    mode = "cricsheet_current_v2"

    try:
        download_zip(url, zip_path, force=force)
        extract_zip(zip_path, raw, force=force)
        matches, players = parse_cricsheet_json(raw, project)
        if matches.empty:
            raise RuntimeError("No binary-result IPL matches parsed from Cricsheet JSON")
    except Exception as exc:
        if not allow_legacy_fallback:
            raise
        mode = "legacy_fallback_v2"
        matches = load_legacy_ipl(project_root / "data/legacy/matches.csv")
        players = pd.DataFrame(columns=[
            "date", "season", "match_id", "team", "player", "appeared", "runs", "balls", "fours", "sixes",
            "dismissals", "wickets", "runs_conceded", "balls_bowled", "dots_bowled"
        ])
        dump_json({"warning": str(exc), "mode": mode}, meta / "download_warning.json")

    matches.to_csv(proc / "matches.csv", index=False)
    players.to_csv(proc / "player_match_stats.csv", index=False)
    report = validate_matches(matches, players)
    report["mode"] = mode
    if zip_path.exists():
        report["raw_zip_sha256"] = sha256_file(zip_path)
    dump_json(report, meta / "data_validation.json")

    registry = pd.DataFrame([
        {
            "source": "Cricsheet",
            "url": url,
            "dataset": "IPL ball-by-ball JSON",
            "retrieved_or_checked_at": pd.Timestamp.utcnow().isoformat(),
            "mode": mode,
            "coverage_start": report.get("date_min"),
            "coverage_end": report.get("date_max"),
            "sha256": report.get("raw_zip_sha256", ""),
            "notes": "Primary structured source. Legacy fallback is used only when the download is unavailable.",
        },
        {
            "source": "Original user project",
            "url": "https://github.com/unit-mole/IPL-WIN-Prediction",
            "dataset": "legacy matches.csv + notebook",
            "retrieved_or_checked_at": pd.Timestamp.utcnow().isoformat(),
            "mode": "legacy_reference",
            "coverage_start": "2008",
            "coverage_end": "2024",
            "sha256": "",
            "notes": "Preserved for audit/history; never preferred over current Cricsheet data.",
        },
    ])
    registry.to_csv(meta / "source_registry.csv", index=False)
    return matches, players, report
