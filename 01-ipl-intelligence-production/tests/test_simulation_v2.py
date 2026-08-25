from cricket_intel.simulation import active_ipl_teams


def test_active_teams_prefers_current_season_list():
    snap={"active_teams":[f"T{i}" for i in range(10)],"teams":{f"T{i}":{"last_date":"2026-05-01","elo":1500+i} for i in range(10)}}
    assert len(active_ipl_teams(snap))==10
