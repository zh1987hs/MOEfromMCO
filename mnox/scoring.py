from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ScoringResult:
    """Scoring result container."""

    scored: pd.DataFrame
    feature_importance: pd.DataFrame | None
    scoring_mode: str


FEATURE_COLUMNS = [
    "embedding_similarity",
    "positive_support_score",
    "local_density_score",
    "novelty_score",
    "mmseqs_best_fident",
    "mmseqs_best_qcov",
    "mmseqs_best_tcov",
    "mmseqs_best_bits",
    "hmm_best_bitscore",
]


def _prepare_x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[cols].fillna(0.0)


def apply_heuristic_scorer(df: pd.DataFrame, retrieval_cfg: dict[str, Any], variant: str = "default") -> ScoringResult:
    """Apply heuristic scorer with conservative novelty weighting."""
    if df.empty:
        return ScoringResult(df.copy(), None, "heuristic")

    hcfg = retrieval_cfg.get("heuristic", {})
    # backward-compatible fallback
    w_aff = float(hcfg.get("w_affinity", retrieval_cfg.get("w_affinity", 0.75)))
    w_sup = float(hcfg.get("w_positive_support", 0.15))
    w_den = float(hcfg.get("w_local_density", retrieval_cfg.get("w_local_support", 0.08)))
    w_nov = float(hcfg.get("w_novelty", retrieval_cfg.get("w_novelty", 0.02)))

    if variant == "no_novelty":
        w_nov = 0.0
    if variant == "no_support":
        w_sup = 0.0

    z = df.copy()
    z["final_score"] = (
        w_aff * z["embedding_similarity"].fillna(0.0)
        + w_sup * z["positive_support_score"].fillna(0.0)
        + w_den * z["local_density_score"].fillna(0.0)
        + w_nov * z["novelty_score"].fillna(0.0)
    )
    z = z.sort_values("final_score", ascending=False).reset_index(drop=True)
    z["rank"] = np.arange(1, len(z) + 1)
    z["scoring_mode"] = f"heuristic_{variant}"

    # confidence tiers for experiment triage
    q1 = z["final_score"].quantile(0.67)
    q2 = z["final_score"].quantile(0.33)
    z["confidence_tier"] = np.where(
        z["final_score"] >= q1,
        "high",
        np.where(z["final_score"] >= q2, "medium", "low"),
    )
    return ScoringResult(z, None, f"heuristic_{variant}")


def apply_learned_scorer(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    retrieval_cfg: dict[str, Any],
) -> ScoringResult:
    """Train fold-local logistic model and score candidates, with safe fallback handled by caller."""
    lcfg = retrieval_cfg.get("learned", {})
    cols = FEATURE_COLUMNS
    x_train = _prepare_x(train_df, cols)
    x_pred = _prepare_x(pred_df, cols)

    standardize = bool(lcfg.get("standardize_features", True))
    if standardize:
        model: Any = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    else:
        model = LogisticRegression(max_iter=2000, class_weight="balanced")

    # lightweight prioritization-oriented weighting: emphasize supported positives and hard negatives.
    sw = np.ones(len(train_labels), dtype=float)
    if "positive_support_score" in train_df.columns:
        sw = np.where(train_labels == 1, 1.0 + 1.5 * np.clip(train_df["positive_support_score"].fillna(0.0).values, 0.0, 1.0), sw)
    if "local_density_score" in train_df.columns:
        sw = np.where(train_labels == 0, sw * (1.0 + 0.8 * np.clip(train_df["local_density_score"].fillna(0.0).values, 0.0, 1.0)), sw)

    if isinstance(model, Pipeline):
        model.fit(x_train, train_labels, clf__sample_weight=sw)
    else:
        model.fit(x_train, train_labels, sample_weight=sw)
    proba = model.predict_proba(x_pred)[:, 1]

    out = pred_df.copy()
    # keep novelty as small correction in learned mode too
    nov = out.get("novelty_score", pd.Series(np.zeros(len(out)))).fillna(0.0).values
    out["final_score"] = np.clip(0.97 * proba + 0.03 * nov, 0.0, 1.0)
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["scoring_mode"] = "learned_logistic"

    q1 = out["final_score"].quantile(0.67)
    q2 = out["final_score"].quantile(0.33)
    out["confidence_tier"] = np.where(
        out["final_score"] >= q1,
        "high",
        np.where(out["final_score"] >= q2, "medium", "low"),
    )

    fi = None
    try:
        clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
        coefs = clf.coef_.reshape(-1)
        fi = pd.DataFrame({"feature": cols, "coefficient": coefs}).sort_values("coefficient", ascending=False)
    except Exception:
        fi = None

    return ScoringResult(out, fi, "learned_logistic")
