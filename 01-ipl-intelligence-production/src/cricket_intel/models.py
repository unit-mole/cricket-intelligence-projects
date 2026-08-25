from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns
from .metrics import binary_metrics, probability_objective
from .utils import dump_json, seed_everything


class IdentityCalibrator:
    name = "identity"
    def fit(self, p, y):
        return self
    def transform(self, p):
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


class PlattProbabilityCalibrator:
    name = "platt"
    def __init__(self):
        self.model = LogisticRegression(C=10.0, max_iter=2000, random_state=42)
    @staticmethod
    def _x(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p)).reshape(-1, 1)
    def fit(self, p, y):
        self.model.fit(self._x(p), np.asarray(y, dtype=int))
        return self
    def transform(self, p):
        return self.model.predict_proba(self._x(p))[:, 1]


class IsotonicProbabilityCalibrator:
    name = "isotonic"
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    def fit(self, p, y):
        self.model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=int))
        return self
    def transform(self, p):
        return np.asarray(self.model.predict(np.asarray(p, dtype=float)), dtype=float)


def symmetric_calibrate(calibrator, p):
    """Apply calibration while preserving P(A>B)=1-P(B>A)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    a = np.asarray(calibrator.transform(p), dtype=float)
    b = np.asarray(calibrator.transform(1.0 - p), dtype=float)
    return np.clip(0.5 * (a + 1.0 - b), 1e-6, 1 - 1e-6)


class ProbabilityEnsembleV2:
    def __init__(self, models: dict[str, Any], weights: dict[str, float], calibrator, feature_columns_: list[str], mode: str, metadata: dict | None = None):
        self.models = models
        self.weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns_)
        self.mode = str(mode)
        self.metadata = metadata or {}

    def _matrix(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_columns]
        arr = np.asarray(X, dtype=float)
        return pd.DataFrame(arr, columns=self.feature_columns)

    @staticmethod
    def _model_p(model, X):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    def raw_probability(self, X):
        X = self._matrix(X)
        Xrev = -X
        values = []
        weights = []
        for name, model in self.models.items():
            if name not in self.weights or self.weights[name] <= 0:
                continue
            pf = self._model_p(model, X)
            pr = self._model_p(model, Xrev)
            psym = 0.5 * (pf + 1.0 - pr)
            values.append(psym)
            weights.append(self.weights[name])
        if not values:
            raise RuntimeError("No selected ensemble components are available")
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return np.vstack(values).T @ w

    def predict_proba(self, X):
        raw = self.raw_probability(X)
        p = symmetric_calibrate(self.calibrator, raw)
        return np.c_[1.0 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def augment_symmetry(X: pd.DataFrame, y: pd.Series):
    X = X.reset_index(drop=True).astype(float)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    return pd.concat([X, -X], ignore_index=True), pd.concat([y, 1 - y], ignore_index=True)


def predict_model_symmetrized(model, X: pd.DataFrame) -> np.ndarray:
    X = X.astype(float)
    pf = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    pr = np.asarray(model.predict_proba(-X)[:, 1], dtype=float)
    return np.clip(0.5 * (pf + 1.0 - pr), 1e-6, 1 - 1e-6)


def _model_spaces() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic": {"C": [0.03, 0.07, 0.15, 0.3, 0.6, 1.0, 2.0, 5.0]},
        "random_forest": {
            "n_estimators": [400, 700, 1000], "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [3, 5, 8, 12, 18], "max_features": ["sqrt", 0.6, 0.85],
        },
        "extra_trees": {
            "n_estimators": [400, 700, 1000], "max_depth": [4, 6, 8, 10, 14, None],
            "min_samples_leaf": [2, 4, 6, 10, 15], "max_features": ["sqrt", 0.6, 0.85, 1.0],
        },
        "hist_gradient_boosting": {
            "max_iter": [250, 400, 600], "learning_rate": [0.02, 0.035, 0.05, 0.07],
            "max_leaf_nodes": [7, 15, 23, 31], "min_samples_leaf": [15, 25, 40, 60], "l2_regularization": [1.0, 3.0, 7.0, 12.0],
        },
        "gradient_boosting": {
            "n_estimators": [200, 350, 500], "learning_rate": [0.02, 0.035, 0.05],
            "max_depth": [1, 2, 3], "min_samples_leaf": [4, 8, 15, 25], "subsample": [0.7, 0.85, 1.0],
        },
        "xgboost": {
            "n_estimators": [300, 500, 750], "max_depth": [2, 3, 4], "learning_rate": [0.02, 0.035, 0.05],
            "min_child_weight": [3, 6, 10], "subsample": [0.75, 0.9, 1.0], "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [2.0, 5.0, 10.0], "reg_alpha": [0.0, 0.1, 0.5],
        },
        "lightgbm": {
            "n_estimators": [300, 500, 750], "num_leaves": [7, 15, 23, 31], "learning_rate": [0.02, 0.035, 0.05],
            "min_child_samples": [15, 25, 40, 60], "subsample": [0.75, 0.9, 1.0], "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [1.0, 3.0, 7.0], "reg_alpha": [0.0, 0.1, 0.5],
        },
        "catboost": {
            "iterations": [300, 500, 750], "depth": [3, 4, 5, 6], "learning_rate": [0.02, 0.035, 0.05],
            "l2_leaf_reg": [3.0, 7.0, 12.0], "random_strength": [0.2, 0.7, 1.2],
        },
    }


def available_model_names() -> list[str]:
    names = ["logistic", "random_forest", "extra_trees", "hist_gradient_boosting", "gradient_boosting"]
    try:
        import xgboost  # noqa: F401
        names.append("xgboost")
    except Exception:
        pass
    try:
        import lightgbm  # noqa: F401
        names.append("lightgbm")
    except Exception:
        pass
    try:
        import catboost  # noqa: F401
        names.append("catboost")
    except Exception:
        pass
    return names


def build_estimator(name: str, params: dict[str, Any], seed: int = 42, use_gpu: bool = False):
    p = dict(params)
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=float(p.get("C", 0.5)), max_iter=3000, random_state=seed)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight=None, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=-1, class_weight=None, **p)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        kw = dict(p)
        kw.update({"eval_metric": "logloss", "random_state": seed, "n_jobs": -1, "tree_method": "hist"})
        if use_gpu:
            kw["device"] = "cuda"
        return XGBClassifier(**kw)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, verbosity=-1, n_jobs=-1, **p)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            random_seed=seed, verbose=False, allow_writing_files=False,
            loss_function="Logloss", task_type="GPU" if use_gpu else "CPU", **p
        )
    raise KeyError(name)


def make_portable(model):
    # XGBoost trees trained on GPU are portable; CPU prediction avoids the device-mismatch warning in the app.
    try:
        if model.__class__.__module__.startswith("xgboost"):
            model.set_params(device="cpu")
    except Exception:
        pass
    return model


def _season_partitions(df: pd.DataFrame, selection_seasons: int = 2, calibration_seasons: int = 1, test_seasons: int = 2):
    years = sorted(int(y) for y in pd.to_datetime(df.date).dt.year.unique())
    needed = selection_seasons + calibration_seasons + test_seasons + 3
    if len(years) < needed:
        raise ValueError(f"Need at least {needed} seasons; found {len(years)}")
    test_years = years[-test_seasons:]
    calibration_years = years[-(test_seasons + calibration_seasons):-test_seasons]
    selection_years = years[-(test_seasons + calibration_seasons + selection_seasons):-(test_seasons + calibration_seasons)]
    training_years = [y for y in years if y < selection_years[0]]
    return training_years, selection_years, calibration_years, test_years


def _tuning_years(df: pd.DataFrame, training_years: list[int], folds: int, minimum_training_rows: int) -> list[int]:
    eligible = []
    years = pd.to_datetime(df.date).dt.year
    for y in training_years:
        if int((years < y).sum()) >= minimum_training_rows and int((years == y).sum()) >= 35:
            eligible.append(y)
    return eligible[-folds:]


def tune_one_model(name: str, df: pd.DataFrame, cols: list[str], tuning_years: list[int], trials: int, seed: int, use_gpu: bool):
    spaces = _model_spaces()[name]
    candidates = list(ParameterSampler(spaces, n_iter=trials, random_state=seed))
    if name == "logistic":
        # Exhaustive tiny grid is cheap and deterministic.
        candidates = [{"C": c} for c in spaces["C"]]
    rows = []
    best_params = None
    best_obj = math.inf
    years_series = pd.to_datetime(df.date).dt.year

    for idx, params in enumerate(candidates, 1):
        fold_metrics = []
        failed = None
        for y in tuning_years:
            tr = df[years_series < y]
            va = df[years_series == y]
            if len(tr) < 100 or len(va) < 20:
                continue
            try:
                Xtr, ytr = augment_symmetry(tr[cols], tr.target)
                model = build_estimator(name, params, seed=seed, use_gpu=use_gpu)
                model.fit(Xtr, ytr)
                p = predict_model_symmetrized(model, va[cols])
                m = binary_metrics(va.target, p)
                fold_metrics.append({"year": y, **m})
            except Exception as exc:
                failed = str(exc)
                break
        if failed or not fold_metrics:
            rows.append({"model": name, "trial": idx, "params": json.dumps(params, sort_keys=True), "error": failed or "no_valid_folds"})
            continue
        recency = np.arange(1, len(fold_metrics) + 1, dtype=float)
        recency /= recency.sum()
        ll = np.asarray([m["log_loss"] for m in fold_metrics], dtype=float)
        br = np.asarray([m["brier"] for m in fold_metrics], dtype=float)
        ec = np.asarray([m["ece"] for m in fold_metrics], dtype=float)
        obj = float(np.sum(recency * ll) + 0.25 * np.sum(recency * br) + 0.10 * np.sum(recency * ec) + 0.10 * np.std(ll))
        row = {
            "model": name, "trial": idx, "params": json.dumps(params, sort_keys=True), "objective": obj,
            "mean_log_loss": float(np.mean(ll)), "std_log_loss": float(np.std(ll)), "mean_brier": float(np.mean(br)),
            "mean_ece": float(np.mean(ec)), "fold_years": ",".join(str(m["year"]) for m in fold_metrics),
        }
        rows.append(row)
        if obj < best_obj:
            best_obj = obj
            best_params = params
    if best_params is None:
        raise RuntimeError(f"All tuning trials failed for {name}")
    return best_params, pd.DataFrame(rows)


def prequential_model_predictions(df: pd.DataFrame, years: list[int], cols: list[str], name: str, params: dict, seed: int, use_gpu: bool):
    pieces = []
    all_years = pd.to_datetime(df.date).dt.year
    for y in years:
        tr = df[all_years < y]
        te = df[all_years == y]
        if len(te) == 0:
            continue
        Xtr, ytr = augment_symmetry(tr[cols], tr.target)
        model = build_estimator(name, params, seed=seed, use_gpu=use_gpu)
        model.fit(Xtr, ytr)
        p = predict_model_symmetrized(model, te[cols])
        part = te[["match_id", "date", "year", "team1", "team2", "target"]].copy()
        part["probability"] = p
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def optimize_ensemble_weights(y: np.ndarray, pred_matrix: pd.DataFrame, eligible: list[str]) -> dict[str, float]:
    P = pred_matrix[eligible].to_numpy(dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(eligible)
    if n == 1:
        return {eligible[0]: 1.0}

    def objective(w):
        p = np.clip(P @ w, 1e-6, 1 - 1e-6)
        return probability_objective(y, p) + 0.003 * float(np.sum(w * w))

    result = minimize(
        objective, np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raw = np.ones(n) / n
    else:
        raw = np.asarray(result.x, dtype=float)
    raw[raw < 0.03] = 0.0
    if raw.sum() <= 0:
        raw[np.argmax(result.x if result.success else np.ones(n))] = 1.0
    raw = raw / raw.sum()
    return {name: float(w) for name, w in zip(eligible, raw) if w > 0}


def choose_calibration(raw_p: np.ndarray, y: np.ndarray, dates: pd.Series):
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    raw_p = np.asarray(raw_p, dtype=float)[order]
    y = np.asarray(y, dtype=int)[order]
    cut = max(35, int(len(y) * 0.60))
    cut = min(cut, max(len(y) - 20, 1))
    train_p, train_y = raw_p[:cut], y[:cut]
    eval_p, eval_y = raw_p[cut:], y[cut:]
    candidates = [IdentityCalibrator(), PlattProbabilityCalibrator()]
    if len(train_y) >= 50 and len(np.unique(train_y)) == 2:
        candidates.append(IsotonicProbabilityCalibrator())
    rows = []
    best = None
    best_obj = math.inf
    for cal in candidates:
        try:
            cal.fit(train_p, train_y)
            q = symmetric_calibrate(cal, eval_p)
            m = binary_metrics(eval_y, q)
            obj = probability_objective(eval_y, q)
            rows.append({"calibrator": cal.name, "selection_objective": obj, **m})
            if obj < best_obj:
                best_obj = obj
                best = cal.name
        except Exception as exc:
            rows.append({"calibrator": cal.name, "error": str(exc)})
    if best is None:
        best = "identity"
    final = {"identity": IdentityCalibrator, "platt": PlattProbabilityCalibrator, "isotonic": IsotonicProbabilityCalibrator}[best]()
    final.fit(raw_p, y)
    return final, pd.DataFrame(rows)


def _combine_predictions(base: pd.DataFrame, component_frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    raw = np.zeros(len(out), dtype=float)
    for name, weight in weights.items():
        frame = component_frames[name].reset_index(drop=True)
        if len(frame) != len(out) or not np.array_equal(frame.match_id.astype(str).to_numpy(), out.match_id.astype(str).to_numpy()):
            raise RuntimeError(f"Prediction alignment failure for {name}")
        out[f"p_{name}"] = frame.probability.to_numpy(dtype=float)
        raw += float(weight) * out[f"p_{name}"].to_numpy(dtype=float)
    out["raw_probability"] = np.clip(raw, 1e-6, 1 - 1e-6)
    return out


def train_mode_v2(
    feature_csv: Path,
    artifacts: Path,
    reports: Path,
    mode: str,
    seed: int = 42,
    use_gpu: bool = False,
    training_mode: str = "full",
    selection_seasons: int = 2,
    calibration_seasons: int = 1,
    test_seasons: int = 2,
    tuning_folds: int = 4,
    minimum_training_rows: int = 250,
    trials_full: int = 12,
    trials_quick: int = 4,
):
    seed_everything(seed)
    cols = feature_columns(mode)
    df = pd.read_csv(feature_csv, parse_dates=["date"])
    df = df.dropna(subset=cols + ["target"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if "year" not in df:
        df["year"] = df.date.dt.year
    else:
        df["year"] = df["year"].astype(int)

    training_years, selection_years, calibration_years, test_years = _season_partitions(
        df, selection_seasons=selection_seasons, calibration_seasons=calibration_seasons, test_seasons=test_seasons
    )
    tune_years = _tuning_years(df, training_years, tuning_folds, minimum_training_rows)
    if len(tune_years) < 2:
        raise RuntimeError(f"Insufficient tuning folds: {tune_years}")

    trials = trials_full if training_mode == "full" else trials_quick
    tuned_params: dict[str, dict] = {}
    tuning_reports = []
    failed_models: dict[str, str] = {}
    model_names = available_model_names()
    if training_mode == "quick":
        # Smoke-test mode validates the complete V2 orchestration without spending time on every optional challenger.
        model_names = [n for n in ["logistic", "extra_trees", "hist_gradient_boosting"] if n in model_names]
    quick_defaults = {
        "logistic": {"C": 0.3},
        "extra_trees": {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
        "hist_gradient_boosting": {"max_iter": 120, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 3.0},
    }
    for name in model_names:
        try:
            print(f"[{mode}] Tuning {name} on temporal folds {tune_years} ...", flush=True)
            if training_mode == "quick":
                params = quick_defaults[name]
                # Validate the smoke-test configuration on the same temporal folds.
                fold_rows = []
                ys = pd.to_datetime(df.date).dt.year
                for yy in tune_years:
                    trq, vaq = df[ys < yy], df[ys == yy]
                    Xq, yq = augment_symmetry(trq[cols], trq.target)
                    mq = build_estimator(name, params, seed=seed, use_gpu=False)
                    mq.fit(Xq, yq)
                    pq = predict_model_symmetrized(mq, vaq[cols])
                    mm = binary_metrics(vaq.target, pq)
                    fold_rows.append(mm)
                report = pd.DataFrame([{
                    "model": name, "trial": 1, "params": json.dumps(params, sort_keys=True),
                    "objective": float(np.mean([r["log_loss"] + 0.25*r["brier"] for r in fold_rows])),
                    "mean_log_loss": float(np.mean([r["log_loss"] for r in fold_rows])),
                    "std_log_loss": float(np.std([r["log_loss"] for r in fold_rows])),
                    "mean_brier": float(np.mean([r["brier"] for r in fold_rows])),
                    "mean_ece": float(np.mean([r["ece"] for r in fold_rows])),
                    "fold_years": ",".join(map(str, tune_years)),
                }])
            else:
                params, report = tune_one_model(name, df, cols, tune_years, trials, seed, use_gpu)
            tuned_params[name] = params
            tuning_reports.append(report)
        except Exception as exc:
            failed_models[name] = str(exc)
            print(f"[{mode}] SKIP {name}: {exc}", flush=True)

    if not tuned_params:
        raise RuntimeError("No V2 model could be tuned successfully")
    tuning_df = pd.concat(tuning_reports, ignore_index=True) if tuning_reports else pd.DataFrame()
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(reports / f"tuning_results_{mode}.csv", index=False)
    dump_json(tuned_params, artifacts / f"best_params_{mode}.json")

    selection_frames: dict[str, pd.DataFrame] = {}
    selection_rows = []
    base_selection = None
    for name, params in tuned_params.items():
        try:
            pred = prequential_model_predictions(df, selection_years, cols, name, params, seed, use_gpu)
            if base_selection is None:
                base_selection = pred[["match_id", "date", "year", "team1", "team2", "target"]].copy()
            selection_frames[name] = pred
            m = binary_metrics(pred.target, pred.probability)
            selection_rows.append({"model": name, "selection_score": float(m["log_loss"] + 0.25 * m["brier"]), **m})
        except Exception as exc:
            failed_models[f"selection_{name}"] = str(exc)

    selection_df = pd.DataFrame(selection_rows).sort_values("selection_score")
    if selection_df.empty:
        raise RuntimeError("No model produced selection-period predictions")
    best_score = float(selection_df.selection_score.min())
    eligible = selection_df[selection_df.selection_score <= best_score + 0.025].model.tolist()[:5]
    if len(eligible) < min(2, len(selection_df)):
        eligible = selection_df.model.tolist()[:min(3, len(selection_df))]
    pred_matrix = pd.DataFrame({name: selection_frames[name].probability.to_numpy(dtype=float) for name in eligible})
    weights = optimize_ensemble_weights(base_selection.target.to_numpy(), pred_matrix, eligible)
    selection_df["selected"] = selection_df.model.isin(weights)
    selection_df["ensemble_weight"] = selection_df.model.map(weights).fillna(0.0)
    selection_df.to_csv(reports / f"ensemble_selection_{mode}.csv", index=False)
    dump_json(weights, artifacts / f"ensemble_weights_{mode}.json")

    # Calibration is selected on a season that occurs strictly before the final two-season test window.
    calibration_frames = {
        name: prequential_model_predictions(df, calibration_years, cols, name, tuned_params[name], seed, use_gpu)
        for name in weights
    }
    base_cal = next(iter(calibration_frames.values()))[["match_id", "date", "year", "team1", "team2", "target"]].copy()
    cal_combined = _combine_predictions(base_cal, calibration_frames, weights)
    calibrator, cal_report = choose_calibration(cal_combined.raw_probability.to_numpy(), cal_combined.target.to_numpy(), cal_combined.date)
    cal_report.to_csv(reports / f"calibration_comparison_{mode}.csv", index=False)

    # Strict annual walk-forward evaluation: each test season is predicted only from earlier seasons.
    test_parts = []
    test_year_metrics = []
    for y in test_years:
        frames = {
            name: prequential_model_predictions(df, [y], cols, name, tuned_params[name], seed, use_gpu)
            for name in weights
        }
        base = next(iter(frames.values()))[["match_id", "date", "year", "team1", "team2", "target"]].copy()
        combined = _combine_predictions(base, frames, weights)
        combined["probability"] = symmetric_calibrate(calibrator, combined.raw_probability.to_numpy())
        combined["mode"] = mode
        test_parts.append(combined)
        test_year_metrics.append({"year": int(y), "rows": len(combined), **binary_metrics(combined.target, combined.probability)})
    strict_test = pd.concat(test_parts, ignore_index=True)
    strict_metrics = binary_metrics(strict_test.target, strict_test.probability)
    strict_test.to_csv(reports / f"strict_test_predictions_{mode}.csv", index=False)
    pd.DataFrame(test_year_metrics).to_csv(reports / f"strict_test_by_year_{mode}.csv", index=False)

    # Individual components on the exact same strict test rows, for apples-to-apples model comparison.
    comparison = []
    for name in tuned_params:
        try:
            pieces = prequential_model_predictions(df, test_years, cols, name, tuned_params[name], seed, use_gpu)
            m = binary_metrics(pieces.target, pieces.probability)
            comparison.append({"model": name, "selected_weight": float(weights.get(name, 0.0)), **m})
        except Exception as exc:
            comparison.append({"model": name, "selected_weight": 0.0, "error": str(exc)})
    comparison.append({"model": "calibrated_selected_ensemble_v2", "selected_weight": 1.0, **strict_metrics})
    pd.DataFrame(comparison).to_csv(reports / f"model_comparison_{mode}.csv", index=False)

    # Production calibration uses only out-of-time predictions, but may use all now-known historical labels.
    production_years = selection_years + calibration_years + test_years
    prod_component_frames = {
        name: prequential_model_predictions(df, production_years, cols, name, tuned_params[name], seed, use_gpu)
        for name in weights
    }
    prod_base = next(iter(prod_component_frames.values()))[["match_id", "date", "year", "team1", "team2", "target"]].copy()
    prod_oof = _combine_predictions(prod_base, prod_component_frames, weights)
    prod_calibrator = {"identity": IdentityCalibrator, "platt": PlattProbabilityCalibrator, "isotonic": IsotonicProbabilityCalibrator}[calibrator.name]()
    prod_calibrator.fit(prod_oof.raw_probability.to_numpy(), prod_oof.target.to_numpy())
    prod_oof["probability"] = symmetric_calibrate(prod_calibrator, prod_oof.raw_probability.to_numpy())
    prod_oof.to_csv(reports / f"production_calibration_oof_{mode}.csv", index=False)

    # Fit every tuned candidate on all data for reproducible experimentation; only selected components enter the deployable bundle.
    all_candidates = {}
    selected_models = {}
    Xall, yall = augment_symmetry(df[cols], df.target)
    for name, params in tuned_params.items():
        try:
            model = build_estimator(name, params, seed, use_gpu)
            model.fit(Xall, yall)
            model = make_portable(model)
            all_candidates[name] = model
            if name in weights:
                selected_models[name] = model
        except Exception as exc:
            failed_models[f"production_{name}"] = str(exc)

    meta = {
        "project_version": "2.0.0",
        "mode": mode,
        "features": cols,
        "feature_count": len(cols),
        "rows": int(len(df)),
        "date_min": str(df.date.min().date()),
        "date_max": str(df.date.max().date()),
        "training_years": training_years,
        "tuning_years": tune_years,
        "selection_years": selection_years,
        "calibration_years": calibration_years,
        "strict_test_years": test_years,
        "strict_test_metrics": strict_metrics,
        "calibrator": calibrator.name,
        "production_calibrator": prod_calibrator.name,
        "component_weights": weights,
        "best_params": tuned_params,
        "failed_models": failed_models,
        "training_mode": training_mode,
        "gpu_requested": bool(use_gpu),
        "scientific_note": "Strict test predictions are annual walk-forward predictions. Production model is refit on all known matches only after strict evaluation.",
    }
    bundle = ProbabilityEnsembleV2(selected_models, weights, prod_calibrator, cols, mode, metadata=meta)
    joblib.dump(bundle, artifacts / f"{mode}_model_bundle.joblib")
    joblib.dump(all_candidates, artifacts / f"candidate_models_{mode}.joblib")
    dump_json(meta, artifacts / f"model_metadata_{mode}.json")

    if mode == "pretoss":
        joblib.dump(bundle, artifacts / "model_bundle.joblib")
        joblib.dump(all_candidates, artifacts / "candidate_models.joblib")
        dump_json(weights, artifacts / "ensemble_weights.json")
        dump_json(meta, artifacts / "model_metadata.json")

    return pd.DataFrame(comparison), meta


def train_v2(feature_csv: Path, artifacts: Path, reports: Path, config: dict, use_gpu: bool = False, training_mode: str = "full"):
    outputs = {}
    for mode in ["pretoss", "posttoss"]:
        comparison, meta = train_mode_v2(
            feature_csv, artifacts, reports, mode=mode,
            seed=int(config.get("random_seed", 42)),
            use_gpu=use_gpu,
            training_mode=training_mode,
            selection_seasons=int(config.get("selection_seasons", 2)),
            calibration_seasons=int(config.get("calibration_seasons", 1)),
            test_seasons=int(config.get("test_seasons", 2)),
            tuning_folds=int(config.get("tuning_folds", 4)),
            minimum_training_rows=int(config.get("minimum_training_rows", 250)),
            trials_full=int(config.get("tuning_trials_full", 12)),
            trials_quick=int(config.get("tuning_trials_quick", 4)),
        )
        outputs[mode] = {"comparison": comparison, "metadata": meta}
    return outputs


def load_bundle(path: Path):
    return joblib.load(path)
