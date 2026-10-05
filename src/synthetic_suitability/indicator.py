"""Grouped, leakage-safe discovery of interpretable suitability indicators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .config import SyntheticExperimentConfig
from .structural import INDICATOR_FORBIDDEN_COLUMNS


@dataclass
class CandidateResult:
    name: str
    features: list[str]
    estimator: Pipeline
    probabilities: np.ndarray
    metrics: dict[str, float]
    threshold: float
    cost: float
    complexity_rank: int


def fit_indicator(config: SyntheticExperimentConfig) -> dict[str, object]:
    """Discover, evaluate, freeze, and document the structural indicator."""
    out = config.output_dir
    data = pd.read_csv(out / "graph_level_outcomes.csv")
    threshold_pp = config.indicator.primary_success_threshold_pp
    target = f"success_{str(threshold_pp).replace('.', '_')}"
    if target not in data:
        raise KeyError(f"Missing target {target}; rerun outcome aggregation.")
    minimum_observed=min(len(data),max(4,len(data)//2))
    all_features = [f for f in config.indicator.candidate_features if f in data.columns and data[f].notna().sum()>=minimum_observed]
    features = [f for f in config.indicator.external_compatible_features if f in data.columns and data[f].notna().sum()>=minimum_observed]
    forbidden = INDICATOR_FORBIDDEN_COLUMNS.intersection(features)
    if forbidden:
        raise RuntimeError(f"Leakage features requested: {sorted(forbidden)}")
    y = data[target].astype(int).to_numpy()
    groups = data[config.indicator.outer_grouping].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Both success classes are required before fitting an indicator.")
    cv = StratifiedGroupKFold(n_splits=min(5, len(np.unique(groups))), shuffle=True, random_state=config.indicator.random_state)
    candidates = _candidate_models(data, features, config)
    results: list[CandidateResult] = []
    for name, cols, estimator, rank in candidates:
        probabilities = cross_val_predict(
            estimator, data[cols], y, groups=groups, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        decision_threshold = _cost_sensitive_threshold(
            y, probabilities, config.indicator.false_positive_cost, config.indicator.false_negative_cost
        )
        metrics = classification_metrics(y, probabilities, decision_threshold)
        cost = _classification_cost(y, probabilities >= decision_threshold, config)
        results.append(CandidateResult(name, cols, estimator, probabilities, metrics, decision_threshold, cost, rank))
    results.sort(key=lambda r: (r.cost, r.complexity_rank, -r.metrics["precision"]))
    best = results[0]
    frozen = clone(best.estimator).fit(data[best.features], y)
    joblib.dump(frozen, out / "frozen_suitability_indicator.joblib")
    avoid_threshold = _avoid_threshold(y, best.probabilities)
    manifest = _manifest(best, frozen, target, threshold_pp, avoid_threshold, config, data)
    (out / "frozen_suitability_indicator.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    leaderboard = pd.DataFrame([
        {"model": r.name, "features": "|".join(r.features), "threshold": r.threshold, "cost": r.cost, **r.metrics}
        for r in results
    ])
    leaderboard.to_csv(out / "indicator_model_comparison.csv", index=False)
    oof = data[["graph_id", "family", "configuration_id"]].copy()
    oof["target"] = y; oof["probability"] = best.probabilities
    oof["decision"] = _three_region(best.probabilities, avoid_threshold, best.threshold)
    oof.to_csv(out / "indicator_oof_predictions.csv", index=False)
    _robustness_and_importance(data, y, groups, all_features, best, config)
    if config.indicator.leave_one_family_out:
        _family_holdout(data, y, best, config)
    _fit_refinement_indicator(data, features, groups, config)
    _fit_gain_regression(data, features, groups, config)
    return manifest


def _candidate_models(data, features, config):
    settings = config.indicator
    models=[]
    for feature in features:
        models.append((f"single_metric__{feature}", [feature], _pipeline(DecisionTreeClassifier(max_depth=1, min_samples_leaf=settings.min_leaf_graphs, class_weight={0: settings.false_positive_cost, 1: 1.0}, random_state=settings.random_state), scale=False), 0))
    # Limit pair search to metrics that are not entirely missing.
    viable = [f for f in features if data[f].notna().sum() >= max(10, len(data)//2)]
    for first, second in combinations(viable, 2):
        models.append((f"two_metric_logistic__{first}__{second}", [first, second], _pipeline(LogisticRegression(max_iter=2000, class_weight={0: settings.false_positive_cost, 1: 1.0}, random_state=settings.random_state), scale=True), 1))
    models.extend([
        ("shallow_tree", viable, _pipeline(DecisionTreeClassifier(max_depth=settings.max_tree_depth, min_samples_leaf=settings.min_leaf_graphs, class_weight={0: settings.false_positive_cost, 1: 1.0}, random_state=settings.random_state), scale=False), 2),
        ("logistic_weighted_score", viable, _pipeline(LogisticRegression(max_iter=3000, class_weight={0: settings.false_positive_cost, 1: 1.0}, random_state=settings.random_state), scale=True), 3),
    ])
    return models


def _pipeline(model, scale: bool) -> Pipeline:
    steps=[("imputer", SimpleImputer(strategy="median"))]
    if scale: steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _cost_sensitive_threshold(y, probability, fp_cost, fn_cost):
    thresholds=np.unique(np.concatenate(([0.01], probability, [0.99])))
    scored=[]
    for threshold in thresholds:
        pred=probability>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        scored.append(((fp_cost*fp+fn_cost*fn)/len(y), -precision_score(y,pred,zero_division=0), threshold))
    return float(min(scored)[2])


def _avoid_threshold(y, probability):
    candidates=np.unique(np.concatenate(([0.01],probability,[0.99])))
    valid=[]
    for threshold in candidates:
        avoided=probability<=threshold
        if avoided.sum()>=max(5,int(0.05*len(y))):
            missed=float(y[avoided].mean())
            if missed<=0.10: valid.append(threshold)
    return float(max(valid)) if valid else float(np.quantile(probability[y==0],0.25))


def _three_region(probability, avoid, recommend):
    return np.where(probability>=recommend,"Recommend",np.where(probability<=avoid,"Avoid","Uncertain"))


def classification_metrics(y, probability, threshold):
    pred=(probability>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "accuracy": accuracy_score(y,pred), "balanced_accuracy": balanced_accuracy_score(y,pred),
        "precision": precision_score(y,pred,zero_division=0), "recall": recall_score(y,pred,zero_division=0),
        "f1": f1_score(y,pred,zero_division=0), "specificity": tn/max(tn+fp,1),
        "roc_auc": roc_auc_score(y,probability) if len(np.unique(y))==2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "negative_outcome_rate_among_recommended": fp/max(fp+tp,1),
    }


def _classification_cost(y, pred, config):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return float((config.indicator.false_positive_cost*fp+config.indicator.false_negative_cost*fn)/len(y))


def _manifest(best, frozen, target, threshold_pp, avoid_threshold, config, data):
    model=frozen.named_steps["model"]
    payload={
        "status":"FROZEN_ON_SYNTHETIC_ONLY", "model_name":best.name, "features":best.features,
        "target":target, "success_definition_pp":threshold_pp,
        "recommend_threshold":best.threshold, "avoid_threshold":avoid_threshold,
        "uncertain_region":[avoid_threshold,best.threshold], "oof_metrics":best.metrics,
        "false_positive_cost":config.indicator.false_positive_cost,
        "fit_graphs":len(data), "fit_families":sorted(data["family"].unique().tolist()),
        "external_real_data_used_for_fit":False,
        "synthetic_training_fingerprint": hashlib.sha256(
            data[["graph_id", target]].sort_values("graph_id").to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    if isinstance(model,LogisticRegression):
        payload["standardized_coefficients"]=[float(x) for x in model.coef_[0]]
        payload["intercept"]=float(model.intercept_[0])
        payload["imputer_medians"]=[float(x) for x in frozen.named_steps["imputer"].statistics_]
        if "scaler" in frozen.named_steps:
            payload["scaler_mean"]=[float(x) for x in frozen.named_steps["scaler"].mean_]
            payload["scaler_scale"]=[float(x) for x in frozen.named_steps["scaler"].scale_]
        payload["formula_note"]="Apply the saved imputer and scaler, then sigmoid(intercept + sum(coef_i*z_i))."
    elif isinstance(model,DecisionTreeClassifier):
        payload["decision_rule"]=export_text(model,feature_names=best.features)
    return payload


def _robustness_and_importance(data,y,groups,features,best,config):
    out=config.output_dir; rng=np.random.default_rng(config.indicator.random_state)
    forest=_pipeline(RandomForestClassifier(n_estimators=500,min_samples_leaf=max(2,config.indicator.min_leaf_graphs//3),class_weight={0:config.indicator.false_positive_cost,1:1.0},random_state=config.indicator.random_state,n_jobs=1),scale=False)
    forest.fit(data[features],y)
    perm=permutation_importance(forest,data[features],y,n_repeats=20,scoring="balanced_accuracy",random_state=config.indicator.random_state,n_jobs=1)
    directions={}
    for feature in features:
        clean=data[[feature]].copy(); clean[feature]=clean[feature].fillna(clean[feature].median())
        directions[feature]=float(np.corrcoef(clean[feature],data["gain_selected_rewiring"])[0,1]) if float(clean[feature].std())>0 else float("nan")
    rows=[]
    unique_configs=np.unique(groups)
    for i,feature in enumerate(features):
        boot=[]
        for _ in range(config.indicator.bootstrap_repetitions):
            sampled=rng.choice(unique_configs,size=len(unique_configs),replace=True)
            chunks=[data.loc[groups==g,[feature,"gain_selected_rewiring"]] for g in sampled]
            frame=pd.concat(chunks,ignore_index=True); frame=frame.replace([np.inf,-np.inf],np.nan).dropna()
            if len(frame)>2 and frame[feature].std()>0: boot.append(frame[feature].corr(frame["gain_selected_rewiring"],method="spearman"))
        rows.append({"metric":feature,"permutation_importance":perm.importances_mean[i],"importance_std":perm.importances_std[i],"direction_correlation":directions[feature],"bootstrap_direction_stability":float(np.mean(np.sign(boot)==np.sign(directions[feature]))) if boot else float("nan"),"bootstrap_q025":float(np.quantile(boot,.025)) if boot else float("nan"),"bootstrap_q975":float(np.quantile(boot,.975)) if boot else float("nan")})
    pd.DataFrame(rows).sort_values("permutation_importance",ascending=False).to_csv(out/"metric_importance_direction_robustness.csv",index=False)


def _family_holdout(data,y,best,config):
    groups=data["family"].astype(str).to_numpy(); probabilities=np.full(len(data),np.nan)
    for train,test in LeaveOneGroupOut().split(data[best.features],y,groups):
        model=clone(best.estimator).fit(data.iloc[train][best.features],y[train])
        probabilities[test]=model.predict_proba(data.iloc[test][best.features])[:,1]
    metrics=classification_metrics(y,probabilities,best.threshold)
    (config.output_dir/"leave_one_family_out_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")


def _fit_refinement_indicator(data,features,groups,config):
    threshold=config.indicator.primary_success_threshold_pp/100.0; y=(data["gain_vgae"]>threshold).astype(int).to_numpy()
    if len(np.unique(y))<2: return
    cv=StratifiedGroupKFold(n_splits=min(5,len(np.unique(groups))),shuffle=True,random_state=config.indicator.random_state)
    model=_pipeline(LogisticRegression(max_iter=3000,class_weight={0:config.indicator.false_positive_cost,1:1.0},random_state=config.indicator.random_state),scale=True)
    p=cross_val_predict(model,data[features],y,groups=groups,cv=cv,method="predict_proba",n_jobs=1)[:,1]
    t=_cost_sensitive_threshold(y,p,config.indicator.false_positive_cost,config.indicator.false_negative_cost)
    fitted=model.fit(data[features],y); joblib.dump(fitted,config.output_dir/"frozen_vgae_refinement_indicator.joblib")
    payload={"status":"FROZEN_ON_SYNTHETIC_ONLY","features":features,"target":"gain_vgae","success_definition_pp":config.indicator.primary_success_threshold_pp,"recommend_threshold":t,"avoid_threshold":_avoid_threshold(y,p),"oof_metrics":classification_metrics(y,p,t),"external_real_data_used_for_fit":False}
    (config.output_dir/"frozen_vgae_refinement_indicator.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


def _fit_gain_regression(data, features, groups, config):
    """Predict expected gain with grouped out-of-fold evaluation."""
    y=data["gain_selected_rewiring"].to_numpy(dtype=float)
    cv=GroupKFold(n_splits=min(5,len(np.unique(groups))))
    models={
        "ridge":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]),
        "random_forest_exploratory":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=500,min_samples_leaf=max(2,config.indicator.min_leaf_graphs//3),random_state=config.indicator.random_state,n_jobs=1))]),
    }
    rows=[]; predictions={}
    for name,model in models.items():
        pred=cross_val_predict(model,data[features],y,groups=groups,cv=cv,n_jobs=1)
        predictions[name]=pred
        rows.append({"model":name,"mae":mean_absolute_error(y,pred),"rmse":mean_squared_error(y,pred)**0.5,"r2":r2_score(y,pred),"spearman":pd.Series(y).corr(pd.Series(pred),method="spearman")})
    pd.DataFrame(rows).to_csv(config.output_dir/"gain_regression_metrics.csv",index=False)
    output=data[["graph_id","family","configuration_id"]].copy(); output["observed_gain"]=y
    for name,pred in predictions.items(): output[f"predicted_{name}"]=pred
    output.to_csv(config.output_dir/"gain_regression_oof_predictions.csv",index=False)
    final=models["ridge"].fit(data[features],y); joblib.dump(final,config.output_dir/"frozen_gain_regressor.joblib")


def validate_on_real_datasets(config: SyntheticExperimentConfig, metrics_csv: Path, gains_csv: Path) -> pd.DataFrame:
    """Apply the already frozen indicator to real data without refitting."""
    model=joblib.load(config.output_dir/"frozen_suitability_indicator.joblib")
    manifest=json.loads((config.output_dir/"frozen_suitability_indicator.json").read_text(encoding="utf-8"))
    metrics=pd.read_csv(metrics_csv); gains=pd.read_csv(gains_csv)
    original=metrics[metrics["graph_name"].eq("Original")].copy() if "graph_name" in metrics else metrics.copy()
    data=original.merge(gains[["dataset","corrected_mean_gain"]],on="dataset",how="left")
    p=model.predict_proba(data[manifest["features"]])[:,1]
    data["suitability_probability"]=p
    data["frozen_decision"]=_three_region(p,manifest["avoid_threshold"],manifest["recommend_threshold"])
    threshold=manifest["success_definition_pp"]/100.0; data["observed_success"]=(data["corrected_mean_gain"]>threshold).astype(int)
    data.to_csv(config.output_dir/"external_real_dataset_validation.csv",index=False)
    metrics_out=classification_metrics(data["observed_success"].to_numpy(),p,manifest["recommend_threshold"])
    (config.output_dir/"external_real_dataset_metrics.json").write_text(json.dumps(metrics_out,indent=2),encoding="utf-8")
    return data
