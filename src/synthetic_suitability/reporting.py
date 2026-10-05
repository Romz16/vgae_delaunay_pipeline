"""Paper-ready figures, tables, and academic text for the synthetic study."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import SyntheticExperimentConfig


def generate_outputs(config: SyntheticExperimentConfig) -> None:
    """Generate requested figures, tables, and manuscript subsections."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("matplotlib and seaborn are required for paper figures") from exc
    out=config.output_dir; figs=out/"figures"; tables=out/"tables"; text_dir=out/"manuscript"
    figs.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True); text_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(out/"graph_level_outcomes.csv")
    importance=pd.read_csv(out/"metric_importance_direction_robustness.csv")
    oof=pd.read_csv(out/"indicator_oof_predictions.csv")
    manifest=json.loads((out/"frozen_suitability_indicator.json").read_text(encoding="utf-8"))
    sns.set_theme(style="whitegrid",context="paper")
    _scatter(data,"lambda2_norm_laplacian","gain_selected_rewiring",figs/"01_lambda2_vs_rewiring_gain.png","Normalized Laplacian lambda 2","Selected rewiring gain")
    _scatter(data,"approx_avg_shortest_path","gain_selected_rewiring",figs/"02_path_length_vs_gain.png","Average shortest path","Selected rewiring gain")
    _scatter(data,"spectral_sweep_conductance","gain_selected_rewiring",figs/"03_bottleneck_vs_gain.png","Spectral sweep conductance","Selected rewiring gain")
    plt.figure(figsize=(6.5,5)); sc=plt.scatter(data["lambda2_norm_laplacian"],data["approx_avg_shortest_path"],c=100*data["gain_selected_rewiring"],cmap="coolwarm",s=25,alpha=.8); plt.colorbar(sc,label="Gain in macro F1 percentage points"); plt.xlabel("Normalized Laplacian lambda 2"); plt.ylabel("Average shortest path"); plt.tight_layout(); plt.savefig(figs/"04_lambda2_path_gain_map.png",dpi=300); plt.close()
    _structural_region(data,figs/"05_structural_success_region.png")
    plt.figure(figsize=(6,5)); sns.scatterplot(data=data,x="gain_geometry",y="gain_vgae",hue="family",alpha=.75,s=35); plt.axhline(0,color="black",lw=.7); plt.axvline(0,color="black",lw=.7); plt.xlabel("Geometry gain"); plt.ylabel("Additional VGAE gain"); plt.tight_layout(); plt.savefig(figs/"06_geometry_gain_vs_vgae_gain.png",dpi=300); plt.close()
    _calibration(oof,figs/"07_success_probability_vs_score.png")
    real_path=out/"external_real_dataset_validation.csv"
    if real_path.exists(): _synthetic_real_space(data,pd.read_csv(real_path),figs/"08_synthetic_and_real_structural_space.png")
    importance.to_csv(tables/"metric_importance_direction_robustness.csv",index=False)
    pd.read_csv(out/"indicator_model_comparison.csv").to_csv(tables/"indicator_model_comparison.csv",index=False)
    _write_academic_sections(config,data,manifest,text_dir)
    _write_analysis_summary(config,data,manifest,importance,out/"DETAILED_ANALYSIS.md")


def _scatter(data,x,y,path,xlabel,ylabel):
    import matplotlib.pyplot as plt
    import seaborn as sns
    frame=data[[x,y,"family"]].replace([np.inf,-np.inf],np.nan).dropna()
    plt.figure(figsize=(6.4,4.5)); sns.scatterplot(data=frame,x=x,y=y,hue="family",alpha=.75,s=32); sns.regplot(data=frame,x=x,y=y,scatter=False,color="black",line_kws={"lw":1}); plt.axhline(0,color="gray",lw=.7); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()


def _structural_region(data,path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    frame=data[["lambda2_norm_laplacian","approx_avg_shortest_path","gain_selected_rewiring"]].dropna().copy()
    frame["lambda_bin"]=pd.qcut(frame["lambda2_norm_laplacian"],q=min(8,frame["lambda2_norm_laplacian"].nunique()),duplicates="drop")
    frame["path_bin"]=pd.qcut(frame["approx_avg_shortest_path"],q=min(8,frame["approx_avg_shortest_path"].nunique()),duplicates="drop")
    pivot=frame.pivot_table(index="path_bin",columns="lambda_bin",values="gain_selected_rewiring",aggfunc="mean",observed=True)
    plt.figure(figsize=(8,5)); sns.heatmap(100*pivot,cmap="coolwarm",center=0,cbar_kws={"label":"Mean gain in macro F1 percentage points"}); plt.xlabel("lambda 2 quantile"); plt.ylabel("path length quantile"); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()


def _calibration(oof,path):
    import matplotlib.pyplot as plt
    frame=oof.copy(); frame["bin"]=pd.qcut(frame["probability"],q=min(10,frame["probability"].nunique()),duplicates="drop")
    agg=frame.groupby("bin",observed=True).agg(score=("probability","mean"),observed=("target","mean"),n=("target","size")).reset_index()
    plt.figure(figsize=(5.5,4.5)); plt.plot([0,1],[0,1],"--",color="gray"); plt.plot(agg["score"],agg["observed"],"o-"); plt.xlabel("Predicted suitability probability"); plt.ylabel("Observed success probability"); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()


def _synthetic_real_space(synthetic,real,path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5,5)); plt.scatter(synthetic["lambda2_norm_laplacian"],synthetic["approx_avg_shortest_path"],s=18,alpha=.25,label="Synthetic"); plt.scatter(real["lambda2_norm_laplacian"],real["approx_avg_shortest_path"],s=55,marker="D",label="Real");
    for _,r in real.iterrows(): plt.annotate(str(r["dataset"]),(r["lambda2_norm_laplacian"],r["approx_avg_shortest_path"]),fontsize=7)
    plt.xlabel("Normalized Laplacian lambda 2"); plt.ylabel("Average shortest path"); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()


def _write_academic_sections(config,data,manifest,text_dir):
    text_dir.joinpath("Structural_Rewiring_Suitability.md").write_text(f"""## Structural Rewiring Suitability

We investigated whether properties of the original unlabeled topology predict the downstream effect of task-informed geometric rewiring. The discovery dataset contained {len(data)} independent graph instances drawn from {data['family'].nunique()} generative families. Pipeline seeds, feature realizations, and repeated instances were treated as nested repetitions rather than independent observations. All candidate predictors were computed before rewiring. Baseline predictive performance, test labels, post-rewiring metrics, and test gains were excluded from the indicator features.

Model development used grouped cross-validation by structural configuration, with leave-one-family-out evaluation as an additional distribution-shift test. False-positive recommendations received a cost of {config.indicator.false_positive_cost:g}, compared with {config.indicator.false_negative_cost:g} for false negatives. The frozen indicator uses {manifest['model_name']} and predicts whether the mean macro-F1 gain exceeds {manifest['success_definition_pp']:g} percentage points. Its three-way decision policy uses Avoid below {manifest['avoid_threshold']:.4f}, Recommend above {manifest['recommend_threshold']:.4f}, and Uncertain between these thresholds. Formula selection and thresholds were frozen before evaluation on real datasets.
""",encoding="utf-8")
    text_dir.joinpath("When_Does_VGAE_Refinement_Help.md").write_text("""## When Does VGAE Refinement Help

We decomposed the total effect into a geometric reconstruction gain and an additional VGAE-refinement gain. For every nested run, the geometric contribution was defined as the test macro-F1 of learned Delaunay minus the original-graph score. The refinement contribution was defined as the test macro-F1 of the validation-selected positive refinement rate minus learned Delaunay at zero percent refinement. This decomposition prevents improvements caused by the geometric scaffold from being attributed to VGAE. A second indicator was fitted using only prerewiring structural metrics and was frozen independently of the real datasets.
""",encoding="utf-8")
    text_dir.joinpath("Limitations.md").write_text("""## Limitations

Synthetic families provide controlled interventions but cannot reproduce every dependency between topology, attributes, and labels found in empirical graphs. Structural variables remain correlated even under matched designs, so the fitted score is predictive rather than a complete causal estimand. The auxiliary GCN makes the treatment task-informed, whereas the indicator is intentionally topology-only; irreducible uncertainty is therefore expected when feature quality varies. External validation contains only twelve real datasets and is underpowered for refitting or threshold revision. The frozen rule must not be adjusted after inspecting those outcomes. Computational conclusions also require measurements on identical hardware and software environments.
""",encoding="utf-8")


def _write_analysis_summary(config,data,manifest,importance,path):
    top=importance.head(8)
    lines=["# Synthetic Structural Rewiring Suitability Analysis","",f"Graphs analyzed: {len(data)}",f"Families: {data['family'].nunique()}",f"Configurations: {data['configuration_id'].nunique()}","", "## Frozen indicator","",f"Model: {manifest['model_name']}",f"Features: {', '.join(manifest['features'])}",f"Recommend threshold: {manifest['recommend_threshold']:.4f}",f"Avoid threshold: {manifest['avoid_threshold']:.4f}","", "## Highest exploratory importances",""]
    for _,row in top.iterrows(): lines.append(f"- {row['metric']}: importance={row['permutation_importance']:.4f}, direction={row['direction_correlation']:.4f}, robustness={row['bootstrap_direction_stability']:.3f}")
    lines.extend(["","## Interpretation rule","","Post-rewiring metrics explain mechanisms but are excluded from the indicator. Real-dataset outcomes are external validation only and must never be used to modify the frozen model or thresholds."])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
