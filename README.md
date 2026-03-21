# MnOx MCO Candidate Prioritization Pipeline

This repository provides **a remote-homology-guided candidate prioritization pipeline for putative Mn(II)-oxidizing MCO discovery**. It is designed to help users prioritize candidate proteins for downstream experimental follow-up rather than to assign definitive function. The workflow supports both high-confidence neighbor expansion and remote discovery in low-identity sequence space. It also includes family-aware cross-validation and a ranking robustness analysis tool for post hoc inspection of remote-only candidates.

## 1. What this repository does

This pipeline supports:

- **High-confidence neighbor expansion** using easy-hit logic from sequence-homology tools.
- **Remote discovery** in low-identity sequence space using explicit sequence gates plus embedding-based ranking.
- **Family-aware evaluation** to reduce leakage between closely related positive groups.
- **Candidate ranking robustness analysis** for remote-only candidates using repeated weight perturbation.

## 2. Repository structure

| Path | What it does |
|---|---|
| `run_pipeline.py` | Main entry point for the end-to-end pipeline. |
| `config.yaml` | Default configuration for inputs, embedding, clustering, retrieval, remote discovery, exports, and CV. |
| `mnox/` | Core implementation package: embedding, clustering, MMseqs/HMM wrappers, feature construction, retrieval, scoring, CV, plotting, and utilities. |
| `configs/examples/` | Example configurations for common modes such as auto clustering and fixed family-aware evaluation. |
| `scripts/stability_remote_candidates.py` | Recommended post-processing CLI for remote-only candidate ranking robustness analysis. |
| `tests/` | Lightweight tests for retrieval logic and stability-analysis behavior. |

## 3. Required input files

| File | Required? | Purpose | Expected format / key columns |
|---|---|---|---|
| Positive FASTA | Required | Positive reference set used for clustering, homology search, support features, and evaluation. | FASTA; sequence identifiers must be stable across optional metadata and fixed-cluster files. |
| Unlabeled MCO FASTA | Required | Candidate search space to rank. | FASTA of unlabeled MCO-like proteins. |
| Positive metadata CSV | Optional, strongly recommended | Supplies tier labels and family labels for interpretation and family-aware CV. | CSV with `positive_id,tier,gold_family,seed_gold_id`. |
| Fixed positive cluster CSV | Optional | Supplies pre-defined positive clusters for fixed-cluster workflows or model-comparison studies. | CSV with `positive_id,cluster`. |

Typical default filenames in `config.yaml` are:

- `positives.fasta`
- `unlabeled_mco.fasta`
- optional metadata CSV
- optional `positive_clusters_fixed.csv`

## 4. Input preparation

### Gold positives
Gold positives should be the strongest experimentally supported MnOx-related MCO sequences available to your study. These are the main anchors used for biological interpretation.

### High-confidence / expanded positives
You may include additional high-confidence homologs as silver or expanded positives to broaden support and improve retrieval stability, but they should be clearly labeled in the metadata.

### Positive metadata CSV
Recommended columns:

```csv
positive_id,tier,gold_family,seed_gold_id
```

- `positive_id`: must match the positive FASTA identifier exactly.
- `tier`: usually `gold` or `silver`.
- `gold_family`: family label used for family-aware CV.
- `seed_gold_id`: the original gold anchor associated with the sequence.

### Fixed cluster CSV
If you want fixed positive clusters, prepare:

```csv
positive_id,cluster
```

This is most useful when comparing embedding models or when you want the clustering scheme to stay fixed across runs.

## 5. Quick start

### Minimal main pipeline command

```bash
python run_pipeline.py
```

Outputs are written to a timestamped subdirectory under the configured output base directory (default: `runs/`).

### Remote discovery example

Make sure remote discovery is enabled and remote-only is the main experimental view:

```yaml
remote_discovery:
  enabled: true
  identity_max: 0.30
  qcov_min: 0.60
  tcov_min: 0.60
  require_missed_only: true

retrieval:
  experimental_view_mode: remote_only
```

Then run:

```bash
python run_pipeline.py
```

### Model comparison example

A practical model-comparison setup is:

- `positive_clustering.mode: fixed`
- `evaluation.mode: leave_one_gold_family_out`
- vary `esm.model_name`

Example:

```bash
python run_pipeline.py
```

In our current use case, a 150M-scale ESM2 model was a strong practical choice for comparison, but that should be treated as an example rather than a universal recommendation.

## 6. Remote discovery workflow

Remote discovery in this repository follows a simple interpretation:

1. **Sequence-homology gates define the remote candidate space.**<br>
   Identity and coverage thresholds decide which candidates are considered truly remote.

2. **ESM2 embeddings prioritize candidates within that remote space.**<br>
   Once the gate is fixed, embedding similarity and related support features rank candidates inside the remote-only pool.

3. **Remote-only outputs are the recommended view for low-identity candidate discovery.**<br>
   If `remote_discovery.enabled=true` and `retrieval.experimental_view_mode=remote_only`, the remote-only outputs are the primary files for remote candidate prioritization.

## 7. Main output files and how to interpret them

### Files mainly used for experimental candidate prioritization

#### `ranked_candidates_remote_only.csv`
The main remote-only ranked table. Use it to inspect every candidate that passed the remote discovery gate.

#### `ranked_candidates_remote_experimental_view.csv`
A more experiment-facing remote-only table. This is usually the most convenient file for selecting remote candidates for follow-up.

#### `top_remote_candidates.csv`
Top-N remote-only candidates in a compact CSV for quick review.

#### `top_remote_candidates.fasta`
FASTA export of the top remote-only candidates for downstream manual inspection or additional sequence analysis.

### Files mainly used for evaluation

#### `cv_summary.csv`
Cross-validation results across methods and views. For remote discovery, pay particular attention to:

- `evaluation_view = remote_only`

#### `ablation_summary.csv`
Mean performance summaries by method, scoring mode, and evaluation view. Useful for comparing heuristic, learned, and ablated ranking variants.

### Files mainly used to understand the remote candidate space

#### `remote_discovery_diagnostics.json`
Summary diagnostics for the remote pool, including how many candidates passed or failed the remote filters.

#### `remote_identity_bins.csv`
Identity-bin summary for the remote candidate space. Useful for checking whether behavior changes across identity ranges.

#### `remote_only_summary_by_family.csv`
Family-level remote-only summary that helps compare how methods behave on families with different amounts of remote holdout signal.

## 8. Candidate ranking robustness analysis

The recommended post-processing step for remote-only candidates is:

```bash
python scripts/stability_remote_candidates.py --run-dir runs/<timestamp>
```

### What the script does
It perturbs the remote-score weights around the current tuned baseline and measures whether target-family candidates remain near the top across repeated reranking.

### Required input
By default it reads:

```text
<run-dir>/ranked_candidates_remote_only.csv
```

### Example command

Default `mcoA-like` analysis:

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family mcoA
```

Analyze another family:

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family cotA_like \
  --n-iter 300 \
  --topk-list 10,20,50,100
```

### How to interpret the main columns

- `top10_freq`, `top20_freq`, `top50_freq`: fraction of perturbation runs in which the candidate stays inside top-K.
- `mean_rank`: average rank across perturbation runs.
- `std_rank`: ranking stability; lower values indicate more stable rankings.
- `best_rank`, `worst_rank`: optimistic and pessimistic rank bounds across the perturbation runs.

Why this is useful:

- a candidate that remains near the top under many small weight perturbations is usually a safer experimental priority than one that ranks high only under a narrow parameter setting;
- stability analysis helps distinguish robust `mcoA-like` remote candidates from candidates that are highly sensitive to a single tuned scoring profile.

## 9. Recommended analysis workflow

A practical first-pass workflow is:

1. Prepare positive FASTA, unlabeled MCO FASTA, and optional metadata / fixed clusters.
2. Run the main pipeline:

```bash
python run_pipeline.py
```

3. Inspect the remote-only outputs:
   - `ranked_candidates_remote_only.csv`
   - `ranked_candidates_remote_experimental_view.csv`
   - `top_remote_candidates.csv`
4. Compare CV results in `cv_summary.csv` and `ablation_summary.csv`, especially `evaluation_view = remote_only`.
5. Run the robustness analysis:

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family mcoA
```

6. Select candidates for experimental validation using both the remote-only ranking and the robustness summary.

## 10. Interpretation notes

- Top-ranked candidates are **not** definitive annotations.
- Merged ranking and remote-only ranking answer different questions and should not be interpreted interchangeably.
- Robustness analysis is for experimental prioritization, not proof of function.
- Family-aware CV should be interpreted separately from the final discovery ranking.

## 11. Optional advanced configuration

Important config keys to know without overloading the first-time user:

- `positive_clustering.mode`: `auto`, `fixed`, or `none`
- `retrieval.scorer`: `heuristic` or `learned`
- `retrieval.experimental_view_mode`: `merged`, `missed_only`, `split_outputs`, or `remote_only`
- `evaluation.mode`: `loco` or `leave_one_gold_family_out`
- `remote_discovery.*`: remote gate and remote ranking settings
- `remote_discovery.ranking.*`: remote-score weights used inside the remote-only candidate space

Useful references:

- `config.yaml`
- `configs/examples/config_auto_loco.yaml`
- `configs/examples/config_fixed_gold_family.yaml`

## 12. Limitations

- Performance can vary substantially across families.
- Semantic shifts can occur when changing embedding model scale.
- Remote-only CV is informative for low-identity discovery, but it should not be confused with overall merged-ranking behavior.
- Stability analysis helps identify robust priorities, but experimental validation is still necessary.
