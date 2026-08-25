from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd

@dataclass
class TournamentTrace:
    participants: list[str]
    super_series_teams: list[str]
    super_series_winner: str
    round2_teams: list[str]
    group_a: list[str]
    group_b: list[str]
    super7_teams: list[str]
    semifinalists: list[str]
    semifinal_pairings: list[tuple[str,str]]
    finalists: list[str]
    champion: str
    match_counts: dict
    group_assignment_mode: str

def validate_participants(participants):
    if len(participants) != 14:
        raise ValueError('2027 tournament scenario requires exactly 14 participants')
    if len(set(participants)) != 14:
        raise ValueError('Participant names must be unique')

def official_match_counts():
    return {'super_series':3,'round2':30,'super7':21,'semifinals':2,'final':1,'total':57}

def _standings(teams, games, strength):
    pts=defaultdict(int); wins=defaultdict(int)
    for a,b,w in games:
        pts[w]+=2; wins[w]+=1
    return sorted(teams,key=lambda t:(pts[t],wins[t],strength[t]),reverse=True)

def _allocate_groups(round2):
    # Deterministic scenario allocation only. It must not be presented as an official group draw.
    return list(round2[::2]), list(round2[1::2])

def _play(cache,a,b,rng):
    return a if rng.random() < cache[(a,b)] else b

def build_probability_cache(probability_fn, teams):
    cache={}
    for i,a in enumerate(teams):
        for b in teams[i+1:]:
            p=float(np.clip(probability_fn(a,b),1e-6,1-1e-6))
            cache[(a,b)]=p; cache[(b,a)]=1-p
    return cache

def simulate_one_2027(cache, strength, participants, rng, groups=None):
    validate_participants(participants)
    ordered=list(participants)
    bottom=ordered[11:14]
    ss=[]
    for a,b in itertools.combinations(bottom,2):
        ss.append((a,b,_play(cache,a,b,rng)))
    survivor=_standings(bottom,ss,strength)[0]
    round2=ordered[:11]+[survivor]
    if groups:
        A=list(groups['A']); B=list(groups['B']); mode='configured'
        if set(A+B)!=set(round2) or len(A)!=6 or len(B)!=6:
            raise ValueError('Configured groups must contain exactly the 12 Round-2 teams in two groups of six')
    else:
        A,B=_allocate_groups(round2); mode='scenario_seeded'
    adv=[]; fourth=[]; round2_games=[]
    for group in [A,B]:
        gg=[]
        for a,b in itertools.combinations(group,2):
            gg.append((a,b,_play(cache,a,b,rng)))
        round2_games += gg
        order=_standings(group,gg,strength)
        adv += order[:3]; fourth.append(order[3])
    best4=max(fourth,key=lambda t:strength[t])
    super7=adv+[best4]
    s7=[]
    for a,b in itertools.combinations(super7,2):
        s7.append((a,b,_play(cache,a,b,rng)))
    order7=_standings(super7,s7,strength)
    sf=order7[:4]
    pairs=[(sf[0],sf[3]),(sf[1],sf[2])]
    w1=_play(cache,*pairs[0],rng); w2=_play(cache,*pairs[1],rng)
    champion=_play(cache,w1,w2,rng)
    counts={'super_series':3,'round2':len(round2_games),'super7':len(s7),'semifinals':2,'final':1,'total':3+len(round2_games)+len(s7)+3}
    return TournamentTrace(ordered,bottom,survivor,round2,A,B,super7,sf,pairs,[w1,w2],champion,counts,mode)

def default_scenario_participants(snapshot):
    dated=[]
    for t,v in snapshot.get('teams',{}).items():
        if v.get('last_date'):
            dated.append((t,pd.Timestamp(v['last_date'])))
    if not dated:
        ranked=sorted(snapshot['teams'],key=lambda t:float(snapshot['teams'][t].get('elo',1500)),reverse=True)
        return ranked[:14]
    maxd=max(d for _,d in dated)
    recent=[t for t,d in dated if d>=maxd-pd.Timedelta(days=900)]
    ranked=sorted(recent,key=lambda t:float(snapshot['teams'][t].get('elo',1500)),reverse=True)
    if len(ranked)<14:
        ranked=sorted(snapshot['teams'],key=lambda t:float(snapshot['teams'][t].get('elo',1500)),reverse=True)
    return ranked[:14]

def simulate_world_cup_2027(probability_fn, snapshot, participants=None, n=10000, seed=42, groups=None):
    teams=list(participants or default_scenario_participants(snapshot))
    validate_participants(teams)
    strength={t:float(snapshot['teams'][t].get('elo',1500.0)) for t in teams}
    cache=build_probability_cache(probability_fn,teams)
    rng=np.random.default_rng(seed)
    ss=defaultdict(int); r2=defaultdict(int); s7=defaultdict(int); semi=defaultdict(int); final=defaultdict(int); champ=defaultdict(int)
    match_counts=None; mode=None
    for _ in range(int(n)):
        tr=simulate_one_2027(cache,strength,teams,rng,groups)
        match_counts=tr.match_counts; mode=tr.group_assignment_mode
        for t in tr.super_series_teams: ss[t]+=1
        for t in tr.round2_teams: r2[t]+=1
        for t in tr.super7_teams: s7[t]+=1
        for t in tr.semifinalists: semi[t]+=1
        for t in tr.finalists: final[t]+=1
        champ[tr.champion]+=1
    out=pd.DataFrame([{
        'team':t,'seed_rank':i+1,
        'super_series_probability':ss[t]/n,'round2_probability':r2[t]/n,'super7_probability':s7[t]/n,
        'semifinal_probability':semi[t]/n,'final_probability':final[t]/n,'championship_probability':champ[t]/n
    } for i,t in enumerate(teams)])
    out.attrs['match_counts']=match_counts; out.attrs['group_assignment_mode']=mode
    return out.sort_values('championship_probability',ascending=False).reset_index(drop=True)
