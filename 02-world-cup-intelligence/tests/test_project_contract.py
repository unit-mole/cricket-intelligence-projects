from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def test_project_config():
    c=json.loads((ROOT/'configs/production.json').read_text()); assert c['champion_model']=='V1' and c['tournament_engine']=='V2_2027'
def test_public_folder_name(): assert json.loads((ROOT/'configs/production.json').read_text())['github_folder']=='02-world-cup-intelligence'
def test_hf_name(): assert json.loads((ROOT/'configs/production.json').read_text())['huggingface_space']=='icc-world-cup-intelligence'
