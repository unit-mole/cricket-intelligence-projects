from cricket_intel.data import _innings_summary, clean_team


def test_team_normalization():
    assert clean_team("  Kings XI Punjab ") == "Punjab Kings"
    assert clean_team("Delhi Daredevils") == "Delhi Capitals"


def test_phase_summary_uses_over_sequence():
    innings = {"overs": []}
    for over in range(16):
        innings["overs"].append({
            "over": over,
            "deliveries": [
                {"batter": "A", "bowler": "B", "runs": {"batter": 4, "total": 4}},
                {"batter": "A", "bowler": "B", "runs": {"batter": 0, "total": 0}},
            ],
        })
    s = _innings_summary(innings)
    assert s["pp_balls"] == 12
    assert s["middle_balls"] == 18
    assert s["death_balls"] == 2
    assert s["pp_rr"] > 0
    assert 0 <= s["dot_pct"] <= 1
