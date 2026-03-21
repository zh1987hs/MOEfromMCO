# MnOx MCO Candidate Prioritization Pipeline

**A remote-homology-guided candidate prioritization pipeline for putative Mn(II)-oxidizing MCO discovery.**

This repository provides a reproducible workflow for prioritizing putative Mn(II)-oxidizing multicopper oxidase (MCO) candidates for follow-up experiments. It supports both high-confidence neighbor expansion and remote discovery in low-identity space, and it includes family-aware cross-validation plus a post hoc ranking robustness analysis tool.

The pipeline is intended for **candidate prioritization**, not definitive functional annotation.

## 1. Key capabilities

### High-confidence neighbor expansion
- Detects strong easy-hit candidates using MMseqs2 and HMM support.
- Produces merged, easy-only, and missed-only views for traditional homolog expansion.

### Remote discovery in low-identity space
- Defines a remote candidate space using sequence identity and coverage thresholds.
- Ranks candidates inside that remote-only space using embedding, support, density, novelty, and risk features.
- Exposes remote-only outputs intended for experimental prioritization.

## 2. Repository structure

| Path | Role |
|---|---|
| `run_pipeline.py` | Main end-to-end pipeline driver. |
| `config.yaml` | Default configuration for inputs, embedding, clustering, retrieval, remote discovery, and CV. |
| `mnox/` | Core library code: embedding, clustering, MMseqs/HMM wrappers, feature building, retrieval, scoring, CV, plotting, and utilities. |
| `configs/examples/` | Example configuration files for common workflows such as auto clustering and fixed family-aware CV. |
| `scripts/stability_remote_candidates.py` | CLI tool for remote-only candidate ranking robustness analysis. |
| `tests/` | Lightweight unit and smoke tests for retrieval and stability-analysis logic. |

## 3. Required input files

| File | Required? | Purpose | Expected format / key columns |
|---|---|---|---|
| Positive FASTA | Required | Positive reference set used for retrieval, clustering, and support signals. | FASTA; sequence IDs must be stable across metadata and optional fixed clusters. |
| Unlabeled MCO FASTA | Required | Candidate search space to rank. | FASTA of unlabeled candidate proteins. |
| Positive metadata CSV | Optional but strongly recommended | Supplies tiers and family labels for family-aware CV and interpretation. | CSV with `positive_id,tier,gold_family,seed_gold_id`. |
| Fixed positive cluster CSV | Optional | Supplies pre-defined positive clusters for fair model comparison or fixed cluster analysis. | CSV with `positive_id,cluster`. |

In the default configuration these correspond to:
- `positives.fasta`
- `unlabeled_mco.fasta`
- optional `metadata_csv`
- optional `positive_clusters_fixed.csv`

## 4. Input preparation guidance

### Gold positives
Use experimentally supported MnOx-related MCO sequences as the core positive set. These sequences should be the highest-confidence functional anchors in your study.

### Silver / high-confidence positives
You may include additional high-confidence homologs to broaden positive support and improve retrieval robustness, but keep track of which sequences are gold versus silver in the metadata.

### Positive metadata CSV
Recommended columns:

```csv
positive_id,tier,gold_family,seed_gold_id
```

- `positive_id`: must match FASTA identifiers exactly.
- `tier`: typically `gold` or `silver`.
- `gold_family`: family label used for family-aware CV.
- `seed_gold_id`: original gold anchor associated with the sequence.

### Fixed cluster CSV
If you want fixed positive clusters, provide:

```csv
positive_id,cluster
```

The IDs must match the positive FASTA and, if metadata is provided, the metadata CSV as well.

## 5. Quick start

### Basic run

```bash
python run_pipeline.py
```

### Remote discovery run

Make sure `config.yaml` keeps remote discovery enabled and uses a remote-only experimental view:

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

### Model comparison use case

For model comparison, a practical pattern is:
- use `positive_clustering.mode: fixed`
- use family-aware evaluation (`evaluation.mode: leave_one_gold_family_out`)
- vary the embedding model in `esm.model_name`

Example:

```bash
python run_pipeline.py
```

In our current use case, a 150M-scale ESM2 model can be a strong practical comparison point, but it should be treated as an example rather than a universal recommendation.

## 6. Remote discovery workflow

Remote discovery should be interpreted in three layers:

1. **Sequence-homology gate defines the remote candidate space.**  
   Candidates enter the remote-only space only if they satisfy the configured identity and coverage thresholds.

2. **ESM2 embeddings rank candidates within that remote space.**  
   Once the gate is fixed, embedding similarity and related support features drive prioritization inside the remote-only pool.

3. **Remote-only outputs are the recommended view for remote candidate prioritization.**  
   If `remote_discovery.enabled=true` and `retrieval.experimental_view_mode=remote_only`, the remote-only outputs are the main experiment-facing files.

## 7. Main output files and how to interpret them

### `ranked_candidates_remote_only.csv`
The core remote-only ranked table. Use this to inspect all candidates that passed the remote gate.

### `ranked_candidates_remote_experimental_view.csv`
A more experiment-facing remote-only view with fields such as:
- `remote_rank`
- `candidate_id`
- `nearest_positive_family`
- `best_identity_to_positive`
- support, density, novelty, HMM, and risk features

### `top_remote_candidates.csv` / `top_remote_candidates.fasta`
Top-N remote-only candidates intended for quick review and downstream laboratory planning. `top_n` is controlled by `remote_discovery.export.top_n`.

### `cv_summary.csv`
Cross-validation summary across methods and views. For remote discovery, pay special attention to:
- `evaluation_view = remote_only`

### `ablation_summary.csv`
Mean performance by evaluation view, method, and scoring mode. Useful for comparing heuristic, learned, and ablated variants.

### `remote_discovery_diagnostics.json`
Summary of how many candidates entered the remote pool and how many were filtered by identity, coverage, or easy-hit restrictions.

### `remote_identity_bins.csv`
Identity-bin summary for the remote search space. Useful for checking whether candidate behavior changes across identity bands.

### `remote_only_summary_by_family.csv`
Family-level summary for `remote_only` evaluation. Particularly useful when reviewing:
- `fold_gold_family`
- `mrr`
- `recall@20`, `recall@50`, `recall@100`
- `n_holdout_pos_in_remote_space`
- `remote_evaluable`

## 8. Candidate ranking robustness analysis

The repository includes a CLI for ranking robustness analysis:

```bash
python scripts/stability_remote_candidates.py --run-dir runs/<timestamp>
```

### What it does
It perturbs the remote ranking weights around the current tuned baseline and asks which target-family candidates remain near the top across repeated reranking.

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
- `std_rank`: ranking stability; lower is more stable.
- `best_rank`, `worst_rank`: optimistic and pessimistic ranking bounds under the tested perturbations.

As a practical rule, candidates with strong baseline ranks **and** good stability statistics are usually better first-round experimental choices than candidates that rank high only under a narrow set of weights.

## 9. Interpretation notes

- Top-ranked candidates are **not** definitive functional annotations.
- Merged ranking and remote-only ranking answer different questions and should not be interpreted interchangeably.
- Stability analysis is a tool for experimental prioritization, not functional proof.

## 10. Optional advanced configuration

Keep the following keys in mind:

- `positive_clustering.mode`: `auto`, `fixed`, or `none`
- `retrieval.scorer`: `heuristic` or `learned`
- `retrieval.experimental_view_mode`: `merged`, `missed_only`, `split_outputs`, or `remote_only`
- `evaluation.mode`: `loco` or `leave_one_gold_family_out`
- `remote_discovery.*`: remote gate and remote ranking configuration
- `remote_discovery.ranking.*`: tuned remote-score weights

For advanced use, see:
- `config.yaml`
- `configs/examples/config_auto_loco.yaml`
- `configs/examples/config_fixed_gold_family.yaml`

## 11. Limitations

- Risk scores are heuristic and interpretable, but they are not calibrated functional probabilities.
- The learned scorer is a lightweight prioritization model, not a universal replacement for the heuristic path.
- Remote-only CV is informative for remote discovery, but it should not be confused with overall merged-ranking performance.
- Stability analysis helps identify candidates that are less sensitive to small weight changes, but it does not prove function.

## 12. Minimal run checklist

1. Prepare a positive FASTA and an unlabeled MCO FASTA.
2. Optionally prepare metadata and fixed clusters.
3. Update `config.yaml`.
4. Run:

```bash
python run_pipeline.py
```

5. Inspect:
- `ranked_candidates_remote_only.csv`
- `ranked_candidates_remote_experimental_view.csv`
- `top_remote_candidates.csv`
- `cv_summary.csv` with `evaluation_view = remote_only`

6. Run robustness analysis:

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family mcoA
```
