import sys,platform,importlib
print('Python:',sys.version); print('Platform:',platform.platform())
for m in ['numpy','pandas','scipy','sklearn','xgboost','lightgbm','catboost','gradio','plotly','joblib','pytest']:
    try:
        x=importlib.import_module(m); print(f'{m:12s}:',getattr(x,'__version__','OK'))
    except Exception as e: print(f'{m:12s}: ERROR {e}')
try:
 import xgboost as x; print('XGBoost build info:',x.build_info())
except Exception: pass
