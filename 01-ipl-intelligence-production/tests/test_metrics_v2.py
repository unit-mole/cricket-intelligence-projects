from cricket_intel.metrics import binary_metrics, ece


def test_metrics_are_in_valid_ranges():
    y=[0,1,0,1]
    p=[.2,.8,.3,.7]
    m=binary_metrics(y,p)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["brier"] <= 1
    assert m["log_loss"] >= 0
    assert 0 <= ece(y,p) <= 1
