from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASE_WEIGHTS: dict[str, float] = {
    "w_affinity": 0.45,
    "w_positive_support": 0.25,
    "w_local_density": 0.10,
    "w_novelty": 0.15,
    "w_false_positive_risk": -0.15,
    "w_identity_penalty": -0.05,
    "w_hmm_support": 0.10,
}

DEFAULT_PERTURB_RANGE: dict[str, float] = {k: 0.05 for k in DEFAULT_BASE_WEIGHTS}

REQUIRED_COLUMNS = [
    "candidate_id",
    "nearest_positive_family",
    "best_identity_to_positive",
    "embedding_similarity",
    "positive_support_score",
    "local_density_score",
    "novelty_score",
    "false_positive_risk",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stability analysis for remote-only candidate rankings under remote_score perturbations."
    )
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory containing remote-only outputs.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Remote-only ranking CSV. Defaults to <run-dir>/ranked_candidates_remote_only.csv.",
    )
    parser.add_argument("--target-family", default="mcoA", help="Target nearest_positive_family to analyze.")
    parser.add_argument("--n-iter", type=int, default=200, help="Number of perturbation iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for perturbation sampling.")
    parser.add_argument("--out-csv", default=None, help="Output CSV path for stability summary.")
    parser.add_argument("--out-json", default=None, help="Output JSON path for analysis metadata.")
    parser.add_argument("--topk-list", default="10,20,50", help="Comma-separated top-k list, e.g. 10,20,50.")
    parser.add_argument(
        "--base-weights-json",
        default=None,
        help="Optional JSON string or path specifying baseline remote_score weights.",
    )
    parser.add_argument(
        "--perturb-range-json",
        default=None,
        help="Optional JSON string or path specifying perturbation ranges per weight (or a scalar).",
    )
    return parser.parse_args(argv)


def _load_json_arg(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    path = Path(text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(text)


def _parse_topk_list(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("--topk-list must contain at least one positive integer.")
    vals = sorted(set(vals))
    if any(v <= 0 for v in vals):
        raise ValueError("--topk-list values must be positive integers.")
    return vals


def _normalize_weight_mapping(raw: Any, defaults: dict[str, float]) -> dict[str, float]:
    if raw is None:
        return dict(defaults)
    if not isinstance(raw, dict):
        raise ValueError("Weight JSON must be an object/dict.")
    out = dict(defaults)
    for key, val in raw.items():
        if key not in defaults:
            raise ValueError(f"Unknown weight key: {key}")
        out[key] = float(val)
    return out


def _normalize_perturb_mapping(raw: Any, weight_keys: list[str]) -> dict[str, float]:
    if raw is None:
        return dict(DEFAULT_PERTURB_RANGE)
    if isinstance(raw, (int, float)):
        return {k: float(raw) for k in weight_keys}
    if not isinstance(raw, dict):
        raise ValueError("Perturb range JSON must be a scalar or an object/dict.")
    default_range = float(raw.get("default", 0.05))
    out = {k: default_range for k in weight_keys}
    for key, val in raw.items():
        if key == "default":
            continue
        if key not in weight_keys:
            raise ValueError(f"Unknown perturbation key: {key}")
        out[key] = float(val)
    return out


def load_remote_candidates(input_csv: str | Path) -> pd.DataFrame:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df


def _derive_hmm_support_score(df: pd.DataFrame, evalue_max: float = 1e-3, bitscore_min: float = 0.0) -> pd.Series:
    evalues = pd.to_numeric(df.get("hmm_best_evalue", pd.Series(np.inf, index=df.index)), errors="coerce").fillna(np.inf)
    bits = pd.to_numeric(df.get("hmm_best_bitscore", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    e_score = (-np.log10(evalues.clip(lower=1e-300)) / max(-np.log10(evalue_max), 1e-9)).clip(0.0, 1.0)
    if bitscore_min > 0:
        b_score = (bits / max(bitscore_min, 1.0)).clip(0.0, 1.0)
    else:
        b_score = (bits > 0).astype(float)
    return np.maximum(e_score, b_score)


def prepare_remote_dataframe(df: pd.DataFrame, identity_max: float = 0.30) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "best_identity_to_positive",
        "embedding_similarity",
        "positive_support_score",
        "local_density_score",
        "novelty_score",
        "false_positive_risk",
        "mmseqs_best_qcov",
        "mmseqs_best_tcov",
        "hmm_best_evalue",
        "hmm_best_bitscore",
        "remote_rank",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "hmm_support_score" not in out.columns:
        out["hmm_support_score"] = _derive_hmm_support_score(out)
    else:
        out["hmm_support_score"] = pd.to_numeric(out["hmm_support_score"], errors="coerce").fillna(0.0)

    if "identity_penalty" not in out.columns:
        out["identity_penalty"] = (
            pd.to_numeric(out["best_identity_to_positive"], errors="coerce").fillna(1.0) / max(identity_max, 1e-9)
        ).clip(0.0, 1.0)
    else:
        out["identity_penalty"] = pd.to_numeric(out["identity_penalty"], errors="coerce").fillna(0.0)
    return out


def compute_remote_score_from_weights(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        float(weights["w_affinity"]) * pd.to_numeric(df["embedding_similarity"], errors="coerce").fillna(0.0)
        + float(weights["w_positive_support"]) * pd.to_numeric(df["positive_support_score"], errors="coerce").fillna(0.0)
        + float(weights["w_local_density"]) * pd.to_numeric(df["local_density_score"], errors="coerce").fillna(0.0)
        + float(weights["w_novelty"]) * pd.to_numeric(df["novelty_score"], errors="coerce").fillna(0.0)
        + float(weights["w_hmm_support"]) * pd.to_numeric(df["hmm_support_score"], errors="coerce").fillna(0.0)
        + float(weights["w_false_positive_risk"]) * pd.to_numeric(df["false_positive_risk"], errors="coerce").fillna(0.0)
        + float(weights["w_identity_penalty"]) * pd.to_numeric(df["identity_penalty"], errors="coerce").fillna(0.0)
    )


def rank_remote_candidates(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["remote_score_recomputed"] = compute_remote_score_from_weights(out, weights)
    out = out.sort_values(["remote_score_recomputed", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    out["baseline_rank_recomputed"] = np.arange(1, len(out) + 1)
    return out


def sample_perturbed_weights(
    base_weights: dict[str, float],
    perturb_range: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    return {
        key: float(base_weights[key] + rng.uniform(-float(perturb_range[key]), float(perturb_range[key])))
        for key in base_weights
    }


def analyze_stability(
    df: pd.DataFrame,
    target_family: str,
    n_iter: int,
    seed: int,
    topk_list: list[int],
    base_weights: dict[str, float],
    perturb_range: dict[str, float],
) -> pd.DataFrame:
    prepared = prepare_remote_dataframe(df)
    baseline_ranked = rank_remote_candidates(prepared, base_weights)
    baseline_rank_map = dict(zip(baseline_ranked["candidate_id"], baseline_ranked["baseline_rank_recomputed"]))

    if "remote_rank" in baseline_ranked.columns:
        baseline_remote_rank_map = dict(
            zip(
                baseline_ranked["candidate_id"],
                pd.to_numeric(baseline_ranked["remote_rank"], errors="coerce").fillna(np.nan),
            )
        )
    else:
        baseline_remote_rank_map = {cid: rank for cid, rank in baseline_rank_map.items()}

    target_mask = baseline_ranked["nearest_positive_family"].fillna("").astype(str).eq(str(target_family))
    target_df = baseline_ranked[target_mask].copy()
    if target_df.empty:
        raise ValueError(f"No remote candidates found for target family: {target_family}")

    candidate_ids = target_df["candidate_id"].tolist()
    rank_history: dict[str, list[int]] = {cid: [] for cid in candidate_ids}
    topk_hits: dict[str, dict[int, int]] = {cid: {k: 0 for k in topk_list} for cid in candidate_ids}

    rng = np.random.default_rng(seed)
    for _ in range(int(n_iter)):
        weights = sample_perturbed_weights(base_weights, perturb_range, rng)
        ranked_iter = rank_remote_candidates(prepared, weights)
        rank_map = dict(zip(ranked_iter["candidate_id"], ranked_iter["baseline_rank_recomputed"]))
        for cid in candidate_ids:
            rank = int(rank_map[cid])
            rank_history[cid].append(rank)
            for k in topk_list:
                if rank <= k:
                    topk_hits[cid][k] += 1

    summary_cols = [
        "candidate_id",
        "nearest_positive_family",
        "best_identity_to_positive",
        "embedding_similarity",
        "positive_support_score",
        "local_density_score",
        "novelty_score",
        "false_positive_risk",
    ]
    summary = target_df[summary_cols].copy()
    summary["baseline_remote_rank"] = summary["candidate_id"].map(baseline_remote_rank_map)
    summary["baseline_rank_recomputed"] = summary["candidate_id"].map(baseline_rank_map)
    for k in topk_list:
        summary[f"top{k}_freq"] = summary["candidate_id"].map(lambda cid, kk=k: topk_hits[cid][kk] / max(1, n_iter))
    summary["mean_rank"] = summary["candidate_id"].map(lambda cid: float(np.mean(rank_history[cid])))
    summary["std_rank"] = summary["candidate_id"].map(lambda cid: float(np.std(rank_history[cid])))
    summary["best_rank"] = summary["candidate_id"].map(lambda cid: int(np.min(rank_history[cid])))
    summary["worst_rank"] = summary["candidate_id"].map(lambda cid: int(np.max(rank_history[cid])))
    summary = summary.sort_values(["baseline_rank_recomputed", "candidate_id"], ascending=[True, True]).reset_index(drop=True)
    return summary


def build_metadata(
    run_dir: str | Path,
    input_csv: str | Path,
    target_family: str,
    n_candidates: int,
    n_iter: int,
    seed: int,
    base_weights: dict[str, float],
    perturb_range: dict[str, float],
    output_csv: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "input_csv": str(input_csv),
        "target_family": target_family,
        "n_candidates": int(n_candidates),
        "n_iter": int(n_iter),
        "seed": int(seed),
        "base_weights": base_weights,
        "perturb_range": perturb_range,
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    input_csv = Path(args.input_csv) if args.input_csv else run_dir / "ranked_candidates_remote_only.csv"
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / f"stability_remote_candidates_{args.target_family}.csv"
    out_json = Path(args.out_json) if args.out_json else run_dir / f"stability_remote_candidates_{args.target_family}.json"

    try:
        topk_list = _parse_topk_list(args.topk_list)
        base_weights = _normalize_weight_mapping(_load_json_arg(args.base_weights_json), DEFAULT_BASE_WEIGHTS)
        perturb_range = _normalize_perturb_mapping(_load_json_arg(args.perturb_range_json), list(base_weights.keys()))
        df = load_remote_candidates(input_csv)
        summary = analyze_stability(
            df=df,
            target_family=args.target_family,
            n_iter=args.n_iter,
            seed=args.seed,
            topk_list=topk_list,
            base_weights=base_weights,
            perturb_range=perturb_range,
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)
        metadata = build_metadata(
            run_dir=run_dir,
            input_csv=input_csv,
            target_family=args.target_family,
            n_candidates=len(summary),
            n_iter=args.n_iter,
            seed=args.seed,
            base_weights=base_weights,
            perturb_range=perturb_range,
            output_csv=out_csv,
            output_json=out_json,
        )
        out_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Wrote stability summary: {out_csv}")
        print(f"[OK] Wrote metadata: {out_json}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
