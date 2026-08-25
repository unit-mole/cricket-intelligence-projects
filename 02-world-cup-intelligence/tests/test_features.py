import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cricket_intel.features import FEATURE_COLUMNS,matchup_from_snapshot

def snap():
    base={'matches':10,'wins':5,'form5':.5,'form10':.5,'form20':.5,'runs10':250,'run_rate10':5,'wickets_taken10':8,'wickets_lost10':8,'last_date':'2026-01-01'}
    return {'teams':{'A':dict(base,elo=1550),'B':dict(base,elo=1450)},'current_season':'2026'}
def test_feature_count(): assert len(FEATURE_COLUMNS)==16
def test_matchup_columns(): assert list(matchup_from_snapshot(snap(),'A','B').columns)==FEATURE_COLUMNS
def test_elo_sign_flip(): assert matchup_from_snapshot(snap(),'A','B').elo_diff.iloc[0]==-matchup_from_snapshot(snap(),'B','A').elo_diff.iloc[0]
def test_numeric_finite(): import numpy as np; assert np.isfinite(matchup_from_snapshot(snap(),'A','B').to_numpy()).all()
