from __future__ import annotations
import platform, sys
mods = ["numpy", "pandas", "scipy", "sklearn", "joblib", "xgboost", "lightgbm", "catboost", "gradio", "plotly", "shap"]
print("Python:", sys.version)
print("Platform:", platform.platform())
for name in mods:
    try:
        m = __import__(name)
        print(f"{name:12s}: {getattr(m, '__version__', 'OK')}")
    except Exception as exc:
        print(f"{name:12s}: ERROR - {exc}")
