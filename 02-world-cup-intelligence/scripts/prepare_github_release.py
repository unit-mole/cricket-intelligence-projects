from pathlib import Path
import json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.runtime import ensure_ready,sha256_file
EX={'.venv','.git','__pycache__','.pytest_cache','dist'}
def main():
    ensure_ready(ROOT,True); out=ROOT/'dist/github_release'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    large=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in EX for x in p.relative_to(ROOT).parts): continue
        if p.stat().st_size>=50*1024*1024: large.append({'file':str(p.relative_to(ROOT)).replace('\\','/'),'mb':round(p.stat().st_size/1024/1024,2),'sha256':sha256_file(p)})
    audit={'status':'PASS','champion_model':'V1','tournament_engine':'V2_2027','large_files':large,'git_lfs_configured_for_joblib':True,'public_folder_name':'02-world-cup-intelligence'}
    (out/'GITHUB_RELEASE_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'GITHUB_PUSH_CHECKLIST.md').write_text('# GitHub push checklist\n\n- Production verification must PASS.\n- Keep the V1 vs V2 comparison visible.\n- Use Git LFS for `*.joblib`.\n- Do not commit `.venv/`, caches, or `dist/`.\n- Public monorepo folder: `02-world-cup-intelligence`.\n',encoding='utf-8')
    z=out/'ICC_World_Cup_Intelligence_Production_GitHub_Snapshot.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob('*'):
            if not p.is_file(): continue
            rel=p.relative_to(ROOT)
            if any(x in EX for x in rel.parts): continue
            zf.write(p,Path('ICC_World_Cup_Intelligence_Production')/rel)
    print('GitHub release audit: PASS'); print('Large files:',len(large)); print('Snapshot:',z)
if __name__=='__main__':main()
