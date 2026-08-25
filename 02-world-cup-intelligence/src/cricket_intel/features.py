
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import math, pandas as pd, numpy as np
from .utils import dump_json

FEATURE_COLUMNS = [
 "elo_diff","form5_diff","form10_diff","form20_diff","season_form_diff","venue_form_diff","h2h_advantage",
 "runs10_diff","run_rate10_diff","wickets_taken10_diff","wickets_lost10_diff","chase_form_diff","defend_form_diff",
 "experience_log_diff","knockout_form_diff","rest_days_diff"
]

@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    scores: deque = field(default_factory=lambda: deque(maxlen=10))
    rr: deque = field(default_factory=lambda: deque(maxlen=10))
    wkts_taken: deque = field(default_factory=lambda: deque(maxlen=10))
    wkts_lost: deque = field(default_factory=lambda: deque(maxlen=10))
    chase: deque = field(default_factory=lambda: deque(maxlen=20))
    defend: deque = field(default_factory=lambda: deque(maxlen=20))
    knockout: deque = field(default_factory=lambda: deque(maxlen=20))
    season_results: list = field(default_factory=list)
    last_date: pd.Timestamp | None = None


def _mean(x, default=0.5): return float(np.mean(list(x))) if len(x) else float(default)
def _scoremean(x): return float(np.mean(list(x))) if len(x) else 0.0

def _is_knockout(stage: str, event: str) -> bool:
    s=(str(stage)+" "+str(event)).lower()
    return any(k in s for k in ["final","semi","qualifier","eliminator","playoff","knockout"])

class PointInTimeFeatureBuilder:
    def __init__(self, k: float=24.0):
        self.k=k; self.teams=defaultdict(TeamState); self.venue=defaultdict(lambda: [0,0]); self.h2h=defaultdict(lambda: [0,0,0])
        self.current_season=None

    def _rate(self, team, n):
        x=list(self.teams[team].recent)[-n:]; return _mean(x)
    def _v_rate(self, team, venue):
        w,m=self.venue[(team,venue)]; return (w+2)/(m+4)
    def _h2h(self,a,b):
        key=tuple(sorted([a,b])); a0,b0,m=self.h2h[key]
        if m==0:return 0.0
        wins_a=a0 if a==key[0] else b0
        return 2*((wins_a+1)/(m+2))-1
    def features(self,a,b,venue,date):
        A=self.teams[a]; B=self.teams[b]
        restA=(pd.Timestamp(date)-A.last_date).days if A.last_date is not None else 14
        restB=(pd.Timestamp(date)-B.last_date).days if B.last_date is not None else 14
        return {
            "elo_diff":A.elo-B.elo,
            "form5_diff":self._rate(a,5)-self._rate(b,5),"form10_diff":self._rate(a,10)-self._rate(b,10),"form20_diff":self._rate(a,20)-self._rate(b,20),
            "season_form_diff":_mean(A.season_results)-_mean(B.season_results),"venue_form_diff":self._v_rate(a,venue)-self._v_rate(b,venue),
            "h2h_advantage":self._h2h(a,b),"runs10_diff":_scoremean(A.scores)-_scoremean(B.scores),"run_rate10_diff":_scoremean(A.rr)-_scoremean(B.rr),
            "wickets_taken10_diff":_scoremean(A.wkts_taken)-_scoremean(B.wkts_taken),"wickets_lost10_diff":_scoremean(B.wkts_lost)-_scoremean(A.wkts_lost),
            "chase_form_diff":_mean(A.chase)-_mean(B.chase),"defend_form_diff":_mean(A.defend)-_mean(B.defend),
            "experience_log_diff":math.log1p(A.matches)-math.log1p(B.matches),"knockout_form_diff":_mean(A.knockout)-_mean(B.knockout),
            "rest_days_diff":float(np.clip(restA-restB,-60,60))}

    def update(self,row):
        a,b,w=row.team1,row.team2,row.winner; date=pd.Timestamp(row.date); season=str(row.season)
        if self.current_season!=season:
            for t in self.teams.values(): t.season_results=[]
            self.current_season=season
        A=self.teams[a]; B=self.teams[b]; ya=1 if w==a else 0; yb=1-ya
        expA=1/(1+10**((B.elo-A.elo)/400)); delta=self.k*(ya-expA); A.elo+=delta; B.elo-=delta
        for t,y in [(a,ya),(b,yb)]:
            st=self.teams[t]; st.matches+=1; st.wins+=y; st.recent.append(y); st.season_results.append(y); st.last_date=date
            vw,vm=self.venue[(t,str(row.venue))]; self.venue[(t,str(row.venue))]=[vw+y,vm+1]
            if _is_knockout(getattr(row,"stage",""),getattr(row,"event_name","")): st.knockout.append(y)
        key=tuple(sorted([a,b])); x,y,m=self.h2h[key]
        if ya: x += 1 if a==key[0] else 0; y += 1 if a==key[1] else 0
        else: x += 1 if b==key[0] else 0; y += 1 if b==key[1] else 0
        self.h2h[key]=[x,y,m+1]
        r1=float(getattr(row,"team1_runs",0) or 0); r2=float(getattr(row,"team2_runs",0) or 0)
        o1=float(getattr(row,"team1_overs",0) or 0); o2=float(getattr(row,"team2_overs",0) or 0)
        w1=float(getattr(row,"team1_wickets",0) or 0); w2=float(getattr(row,"team2_wickets",0) or 0)
        if r1 or r2:
            A.scores.append(r1); B.scores.append(r2); A.rr.append(r1/o1 if o1 else 0); B.rr.append(r2/o2 if o2 else 0)
            A.wkts_lost.append(w1); B.wkts_lost.append(w2); A.wkts_taken.append(w2); B.wkts_taken.append(w1)
        # first innings team is whichever appears first in parsed innings only indirectly; use team1/team2 batting order when runs available is not guaranteed.
        # We therefore update generic chase/defend only when toss decision implies batting order; otherwise leave priors neutral.
        toss=getattr(row,"toss_winner",""); dec=str(getattr(row,"toss_decision","")).lower()
        if toss in {a,b} and dec in {"bat","field"}:
            batting_first=toss if dec=="bat" else (b if toss==a else a); chasing=b if batting_first==a else a
            self.teams[chasing].chase.append(1 if w==chasing else 0); self.teams[batting_first].defend.append(1 if w==batting_first else 0)

    def snapshot(self):
        teams={}
        for name,s in self.teams.items():
            teams[name]={"elo":s.elo,"matches":s.matches,"wins":s.wins,"form5":self._rate(name,5),"form10":self._rate(name,10),"form20":self._rate(name,20),
                         "runs10":_scoremean(s.scores),"run_rate10":_scoremean(s.rr),"wickets_taken10":_scoremean(s.wkts_taken),"wickets_lost10":_scoremean(s.wkts_lost),
                         "last_date":str(s.last_date.date()) if s.last_date is not None else None}
        return {"teams":teams,"current_season":self.current_season}


def build_features(matches: pd.DataFrame, out_dir: Path):
    df=matches.copy(); df["date"]=pd.to_datetime(df.date); df=df.sort_values(["date","match_id"])
    builder=PointInTimeFeatureBuilder(); rows=[]
    for r in df.itertuples(index=False):
        if r.winner not in {r.team1,r.team2}: continue
        feat=builder.features(r.team1,r.team2,str(r.venue),r.date)
        feat.update({"match_id":r.match_id,"date":r.date,"season":str(r.season),"team1":r.team1,"team2":r.team2,"venue":str(r.venue),
                     "event_name":getattr(r,"event_name",""),"stage":getattr(r,"stage",""),"target":1 if r.winner==r.team1 else 0})
        rows.append(feat); builder.update(r)
    out=pd.DataFrame(rows); out_dir.mkdir(parents=True,exist_ok=True); out.to_csv(out_dir/"features.csv",index=False)
    dump_json({"feature_columns":FEATURE_COLUMNS},out_dir/"feature_schema.json"); dump_json(builder.snapshot(),out_dir/"latest_team_snapshot.json")
    return out


def matchup_from_snapshot(snapshot: dict, team1: str, team2: str):
    A=snapshot["teams"][team1]; B=snapshot["teams"][team2]
    # Features unavailable without full serialized history use neutral priors at live inference; core strength comes from snapshot differences.
    f={c:0.0 for c in FEATURE_COLUMNS}
    f.update({"elo_diff":A["elo"]-B["elo"],"form5_diff":A["form5"]-B["form5"],"form10_diff":A["form10"]-B["form10"],
              "form20_diff":A["form20"]-B["form20"],"runs10_diff":A["runs10"]-B["runs10"],"run_rate10_diff":A["run_rate10"]-B["run_rate10"],
              "wickets_taken10_diff":A["wickets_taken10"]-B["wickets_taken10"],"wickets_lost10_diff":B["wickets_lost10"]-A["wickets_lost10"],
              "experience_log_diff":math.log1p(A["matches"])-math.log1p(B["matches"])})
    return pd.DataFrame([[f[c] for c in FEATURE_COLUMNS]],columns=FEATURE_COLUMNS)
