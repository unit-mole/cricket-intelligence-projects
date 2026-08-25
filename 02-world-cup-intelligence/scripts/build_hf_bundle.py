from pathlib import Path
import shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from world_cup_production.runtime import ensure_ready

def cp(src,dst): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def main():
    ensure_ready(ROOT,True); out=ROOT/'dist/huggingface_space'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    cp(ROOT/'app.py',out/'app.py'); cp(ROOT/'deployment/requirements_hf.txt',out/'requirements.txt'); cp(ROOT/'deployment/README_HF_TEMPLATE.md',out/'README.md')
    for pkg in ['cricket_intel','world_cup_production']: shutil.copytree(ROOT/'src'/pkg,out/pkg,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in ['artifacts/model_bundle.joblib','artifacts/model_metadata_v1.json','artifacts/PRODUCTION_ASSET_MANIFEST.json','artifacts/PRODUCTION_READY.flag','data/processed/latest_team_snapshot.json','data/processed/feature_schema.json','data/metadata/world_cup_2027_format.json','data/metadata/world_cup_2027_venues.csv','reports/FROZEN_V1_V2_COMPARISON.csv','reports/FROZEN_CHAMPION_METRICS.json','reports/production_permutation_importance.csv','reports/v1_expanding_window_backtest.csv','reports/v1_model_comparison.csv','reports/V2_ACCEPTANCE_REPORT.json']:
        p=ROOT/rel
        if p.exists(): cp(p,out/rel)
    (out/'.gitignore').write_text('__pycache__/\n*.pyc\n.pytest_cache/\n',encoding='utf-8')
    (out/'.gitattributes').write_text('*.joblib filter=lfs diff=lfs merge=lfs -text\n',encoding='utf-8')
    z=ROOT/'dist/ICC_World_Cup_Intelligence_HuggingFace_Space.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob('*'):
            if p.is_file(): zf.write(p,p.relative_to(out))
    print('Hugging Face Space bundle created:'); print(out); print(z)
if __name__=='__main__':main()
