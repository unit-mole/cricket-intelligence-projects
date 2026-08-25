from pathlib import Path
import numpy as np
import pandas as pd
from cricket_intel.features import build_features, matchup_from_snapshot, PRETOSS_FEATURE_COLUMNS, POSTTOSS_FEATURE_COLUMNS


def _matches(second_runs=170):
    base = []
    for i, (date, winner, r1, r2, toss, dec, first) in enumerate([
        ("2024-01-01", "A", 180, 160, "A", "bat", "A"),
        ("2024-01-10", "B", second_runs, 175, "B", "field", "A"),
        ("2024-01-20", "A", 190, 185, "B", "bat", "B"),
    ]):
        row = {
            "match_id": str(i), "date": pd.Timestamp(date), "season": "2024", "team1": "A", "team2": "B", "winner": winner,
            "venue": "V", "event_name": "IPL", "stage": "league", "toss_winner": toss, "toss_decision": dec,
            "first_innings_team": first, "second_innings_team": "B" if first == "A" else "A",
            "team1_players": "A1|A2", "team2_players": "B1|B2",
        }
        for side, runs in [(1, r1), (2, r2)]:
            row.update({
                f"team{side}_runs": runs, f"team{side}_wickets": 6, f"team{side}_overs": 20.0,
                f"team{side}_run_rate": runs/20, f"team{side}_dot_pct": .34, f"team{side}_boundary_pct": .12,
                f"team{side}_pp_rr": 8.0, f"team{side}_middle_rr": 8.3, f"team{side}_death_rr": 10.2,
                f"team{side}_pp_wicket_rate": .3, f"team{side}_middle_wicket_rate": .4, f"team{side}_death_wicket_rate": .7,
            })
        base.append(row)
    return pd.DataFrame(base)


def _players():
    rows=[]
    for mid,date in [("0","2024-01-01"),("1","2024-01-10"),("2","2024-01-20")]:
        for team, players in [("A",["A1","A2"]),("B",["B1","B2"])]:
            for j,p in enumerate(players):
                rows.append({"date":date,"season":"2024","match_id":mid,"team":team,"player":p,"appeared":1,"runs":20+j*10,"balls":15,"fours":2,"sixes":1,"dismissals":1,"wickets":j,"runs_conceded":20,"balls_bowled":18,"dots_bowled":6})
    return pd.DataFrame(rows)


def test_current_match_score_does_not_leak_into_its_own_features(tmp_path: Path):
    p = _players()
    f1 = build_features(_matches(second_runs=170), p, tmp_path / "a")
    f2 = build_features(_matches(second_runs=999), p, tmp_path / "b")
    # Match 1 is the second chronological row. Its pre-match vector must not depend on its own score.
    a = f1.iloc[1][PRETOSS_FEATURE_COLUMNS].to_numpy(dtype=float)
    b = f2.iloc[1][PRETOSS_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(a, b)


def test_snapshot_matchup_is_antisymmetric(tmp_path: Path):
    f = build_features(_matches(), _players(), tmp_path)
    import json
    snap=json.loads((tmp_path/"latest_team_snapshot_v2.json").read_text())
    ab = matchup_from_snapshot(snap,"A","B",venue="V",mode="pretoss")
    ba = matchup_from_snapshot(snap,"B","A",venue="V",mode="pretoss")
    assert np.allclose(ab.to_numpy(), -ba.to_numpy(), atol=1e-9)


def test_posttoss_matchup_is_antisymmetric(tmp_path: Path):
    build_features(_matches(), _players(), tmp_path)
    import json
    snap=json.loads((tmp_path/"latest_team_snapshot_v2.json").read_text())
    ab = matchup_from_snapshot(snap,"A","B",venue="V",mode="posttoss",toss_winner="A",toss_decision="field",lineup1=["A1","A2"],lineup2=["B1","B2"])
    ba = matchup_from_snapshot(snap,"B","A",venue="V",mode="posttoss",toss_winner="A",toss_decision="field",lineup1=["B1","B2"],lineup2=["A1","A2"])
    assert list(ab.columns) == POSTTOSS_FEATURE_COLUMNS
    assert np.allclose(ab.to_numpy(), -ba.to_numpy(), atol=1e-9)
