from pathlib import Path
import shutil,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MONO=ROOT.parent/'cricket-intelligence-projects'
DEST=MONO/'02-world-cup-intelligence'
EX={'.venv','.git','__pycache__','.pytest_cache','dist'}
if not MONO.exists():
    raise SystemExit('Could not find sibling cricket-intelligence-projects folder.')
if not (ROOT/'artifacts/PRODUCTION_READY.flag').exists():
    raise SystemExit('Production is not verified. Run 02_VERIFY_PRODUCTION_PACKAGE.bat first.')
if DEST.exists():
    shutil.rmtree(DEST)
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if any(x in EX for x in rel.parts):
        continue
    d=DEST/rel
    if p.is_dir():
        d.mkdir(parents=True,exist_ok=True)
    elif p.is_file():
        d.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,d)
# Workflows only run from monorepo root.
nested=DEST/'.github'
wf=DEST/'.github/workflows/tests.yml'
rootwf=MONO/'.github/workflows/world-cup-tests.yml'
rootwf.parent.mkdir(parents=True,exist_ok=True)
if wf.exists():
    txt=wf.read_text(encoding='utf-8').replace('name: tests','name: World Cup Production Tests')
    out=[]
    for line in txt.splitlines():
        out.append(line)
        if line.lstrip().startswith('- run:'):
            out.append('        working-directory: 02-world-cup-intelligence')
    rootwf.write_text('\n'.join(out)+'\n',encoding='utf-8')
if nested.exists():
    shutil.rmtree(nested)
# Update root README without deleting unrelated content.
comp=pd.read_csv(ROOT/'reports/FROZEN_V1_V2_COMPARISON.csv')
v1=comp[comp.system=='V1_MODEL_CHAMPION'].iloc[0]
v2=comp[comp.system=='V2_MODEL_CHALLENGER'].iloc[0]
section=(
    '### 02 - ICC World Cup Intelligence Engine\n\n'
    'Production ODI World Cup intelligence system combining the accepted **V1 forecasting model** with the corrected **V2 2027 tournament simulator**. Model selection and simulator selection are documented separately.\n\n'
    f'- Champion model: **V1**\n'
    f'- Tournament engine: **V2 corrected 2027 format (57 matches)**\n'
    f'- Same-window V1 accuracy: **{v1.accuracy:.2%}**\n'
    f'- Same-window V1 ROC-AUC: **{v1.roc_auc:.4f}**\n'
    f'- V2 challenger accuracy: **{v2.accuracy:.2%}**\n'
    f'- V2 challenger ROC-AUC: **{v2.roc_auc:.4f}**\n'
    '- Model decision: V2 rejected by the pre-registered probability objective\n'
    '- Live demo: to be added after Hugging Face deployment\n\n'
    '[Explore Project 02](./02-world-cup-intelligence)\n'
)
rp=MONO/'README.md'
s=rp.read_text(encoding='utf-8') if rp.exists() else '# Cricket Intelligence Projects\n\n## Projects\n'
pat=r'(?ms)^### 02 - ICC World Cup Intelligence Engine\n.*?(?=^### |^## |\Z)'
if re.search(pat,s):
    s=re.sub(pat,section+'\n',s,count=1)
else:
    s=s.rstrip()+'\n\n'+section
rp.write_text(s,encoding='utf-8')
print('Installed Project 02 into:',DEST)
print('Root workflow:',rootwf)
print('Root README updated:',rp)
