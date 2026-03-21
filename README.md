# MnOx MCO candidate prioritization pipeline / MnOx MCO 候选优先级流程

## Project overview / 项目概述

This repository provides a reproducible workflow for prioritizing putative Mn(II)-oxidizing multicopper oxidase (MCO) candidates, with special support for **remote discovery** and experiment-oriented triage.  
本仓库提供一个可复现的 putative Mn(II)-oxidizing multicopper oxidase (MCO) 候选优先级分析流程，重点支持 **remote discovery（远缘发现）** 与实验优先级排序。

The pipeline is designed to improve **top-K experimental selection**, not to provide definitive functional annotation by itself.  
该流程的目标是提升 **top-K 实验候选筛选效率**，而不是直接给出 definitive functional annotation。

Current workflow supports two complementary use cases.  
当前流程支持两类互补目标：

- **Nearest-neighbor expansion / 近邻扩展**: use easy-hit / merged rankings to expand high-confidence close homologs.
- **Remote discovery / 远缘发现**: define a distant candidate space by sequence identity/coverage gates, then rank candidates inside that remote-only space.

## Recommended interpretation framework / 推荐解释框架

For remote discovery, we recommend the following conceptual split.  
对于 remote discovery，我们建议按以下框架理解结果：

1. **Sequence alignment defines the remote candidate space / 序列比对定义 remote 候选空间**  
   Identity and coverage thresholds determine which candidates are considered truly remote.
2. **ESM2 embedding prioritizes candidates inside the remote space / ESM2 embedding 负责 remote 子空间内优先级排序**  
   Embedding similarity and support features help rank distant candidates once the gate is fixed.
3. **Stability analysis identifies robust candidates / 稳健性分析识别稳健候选**  
   Perturbing ranking weights helps find candidates that stay near the top rather than being driven by a single lucky parameter setting.

## Data preparation / 数据准备

Prepare at least the following inputs.  
至少准备以下输入文件：

- `positives.fasta`: known or curated positive sequences.
- `unlabeled_mco.fasta` (or your configured unlabeled FASTA): candidate search space.
- Optional positive metadata CSV with columns:

```csv
positive_id,tier,gold_family,seed_gold_id
```

You can update the paths directly in `config.yaml`.  
你可以直接在 `config.yaml` 中修改这些路径。

## Running the main pipeline / 运行主流程

### Basic run / 基础运行

```bash
python run_pipeline.py
```

### 150M tuned remote discovery example / 150M tuned remote discovery 示例

Update `config.yaml` (or an example config) to use your desired ESM2 model and keep remote discovery enabled, then run:  
将 `config.yaml`（或示例配置）切换到你希望使用的 ESM2 模型，并保持 remote discovery 开启，然后运行：

```bash
python run_pipeline.py
```

For example, you may switch `esm.model_name` to a larger ESM2 checkpoint such as a 150M-scale model and keep:  
例如，你可以将 `esm.model_name` 切换到更大的 ESM2 checkpoint（如 150M 级别），并保持：

```yaml
remote_discovery:
  enabled: true
  identity_max: 0.30
  qcov_min: 0.60
  tcov_min: 0.60
  require_missed_only: true
```

## Remote-only outputs / remote-only 关键输出

When `remote_discovery.enabled=true` and `experimental_view_mode=remote_only`, the primary experiment-facing outputs come from the remote-only channel.  
当 `remote_discovery.enabled=true` 且 `experimental_view_mode=remote_only` 时，主实验输出来自 remote-only 通道。

Key files include:  
关键文件包括：

- `ranked_candidates_remote_only.csv`
- `ranked_candidates_remote_experimental_view.csv`
- `top_remote_candidates.csv`
- `top_remote_candidates.fasta`

Remote discovery is defined as candidates satisfying all of the following.  
remote discovery 的定义是同时满足以下条件的候选：

- `best_identity_to_positive < 0.30`
- coverage thresholds pass (`qcov` and `tcov`)
- search is restricted to missed-space candidates by default

## How to read remote-only CV / 如何查看 remote_only CV

`cv_summary.csv` contains an `evaluation_view` column.  
`cv_summary.csv` 中包含 `evaluation_view` 列。

For remote discovery, focus on rows where:  
对于 remote discovery，重点查看：

- `evaluation_view = remote_only`

You can also inspect `remote_only_summary_by_family.csv` to compare family-level behavior using:  
你也可以查看 `remote_only_summary_by_family.csv`，按 family 比较：

- `mrr`
- `recall@20`
- `recall@50`
- `recall@100`
- `n_holdout_pos_in_remote_space`
- `remote_evaluable`

## Candidate Ranking Robustness Analysis / 候选排序稳健性分析

The repository now ships a reusable CLI tool:  
仓库现在包含一个可复用的正式 CLI 工具：

```bash
python scripts/stability_remote_candidates.py --run-dir runs/<timestamp>
```

It perturbs remote ranking weights around the tuned baseline and summarizes how stable each target-family candidate remains across repeated reranking.  
它会围绕当前 tuned remote baseline 权重做扰动，并汇总目标 family 候选在重复 rerank 中的稳定性。

### Default mcoA-like example / 默认 mcoA-like 示例

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family mcoA
```

### Other family example / 指定其他 family 示例

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family cotA_like \
  --n-iter 300 \
  --topk-list 10,20,50,100
```

### Custom weight example / 自定义权重示例

```bash
python scripts/stability_remote_candidates.py \
  --run-dir runs/<timestamp> \
  --target-family mcoA \
  --base-weights-json '{"w_affinity": 0.45, "w_positive_support": 0.25, "w_local_density": 0.10, "w_novelty": 0.15, "w_false_positive_risk": -0.15, "w_identity_penalty": -0.05, "w_hmm_support": 0.10}'
```

Important output columns / 关键输出列：

- `top10_freq`, `top20_freq`, `top50_freq`: frequency of staying inside top-K
- `mean_rank`: average reranked position
- `std_rank`: ranking stability / volatility
- `best_rank`, `worst_rank`: optimistic and pessimistic bounds

Recommended interpretation for first-round experimental selection.  
面向首批实验候选的推荐解释方式：

- High `top10_freq` or `top20_freq` suggests a candidate is not driven by one fragile weight setting.
- Lower `mean_rank` and lower `std_rank` indicate stronger ranking robustness.
- Candidates with decent baseline rank but very unstable perturbation behavior should be treated more cautiously.

## Interpretation notes / 结果解读建议

- A top-ranked candidate is **not** the same as definitive functional proof.  
  排名前列候选 **不等于** 功能已被证明。
- Do not mix `remote_only` CV with merged leaderboard interpretation.  
  不要把 `remote_only` CV 与 merged 总榜混在一起解释。
- Stability analysis is a prioritization tool for experiment planning, not functional evidence by itself.  
  稳健性分析是实验优先级工具，不是功能证明本身。

## 本次升级（方法学对齐 + hybrid 优化）

- CV 与正式 discovery 流程对齐：先 easy/missed 分流，再 missed hybrid 重排，再合并。
- retrieval 重构为：
  - `mnox/features.py`：特征构建层
  - `mnox/scoring.py`：scorer 层（heuristic + learned logistic）
- 支持 `easy_hit_policy`（默认 `prepend`）。
- novelty 改为小权重校正，不再主导排序。
- 增强实验友好输出：reason / confidence / risk flags / experimental view。
- 支持 family-aware CV 且防泄漏（leave-one-gold-family-out + leakage_guard）。

## 正样本体系

- gold positives：严格实验验证
- silver positives：从 gold 同源扩展得到的高可信正样本

元数据 CSV：

```csv
positive_id,tier,gold_family,seed_gold_id
```

## 聚类模式

`positive_clustering.mode`：
- `fixed`：簇成员固定，prototype 每次随当前 embedding 模型重算（公平比较 35M vs 150M 推荐）
- `auto`：训练内自动聚类
- `none`：全部正样本单簇

## 评估模式

`evaluation.mode`：
- `loco`
- `leave_one_gold_family_out`（推荐严格评估）

family-aware 模式下：
- 每折留出一个 `gold_family`
- `leakage_guard=true` 时，该 family 的 gold+silver 都从训练锚点移除

## retrieval 配置（新）

见 `config.yaml`：

- `retrieval.scorer`: `heuristic | learned`（默认建议 `heuristic` 作为稳定主线，`learned` 作为可选增强）
- `retrieval.easy_hit_policy`: `prepend | merge | interleave | prepend_with_cap`
- `easy_hit.decision_rule`: `or | and | consensus | calibrated`
- `easy_hit.profile`: `loose | balanced | strict`
- `easy_hit.max_easy_fraction`: easy 膨胀上限（推荐 0.30）
- `easy_hit.fallback_if_too_many_easy`: `tighten | demote_to_missed | warn_only`
- `retrieval.support_topk`
- `retrieval.density_knn_k`
- `retrieval.experimental_view_mode`: `merged | missed_only | split_outputs`
- `retrieval.cv_view_mode`: `merged | missed_only | remote_only | both`
- `retrieval.export_easy_only / export_missed_only`
- `retrieval.heuristic.*`
- `retrieval.learned.*`
- `retrieval.risk.*`：风险量化权重
- `retrieval.experimental_priority.*`：实验优先级排名中的风险惩罚权重

### remote discovery（最小补充）

- 当前仓库支持两类输出视图：
  - easy / missed / merged 常规视图
  - remote discovery 远缘发现视图
- remote discovery 指默认在与所有正样本最佳 identity `< 30%` 的远缘空间中筛选候选，并要求覆盖度通过阈值，且默认只在 missed 空间中搜索。
- 当 `remote_discovery.enabled = true` 且 `experimental_view_mode = remote_only` 时，主实验输出默认来自 **remote 榜单**（`ranked_candidates_experimental_view.csv`）。
- remote-only 主输出文件：
  - `ranked_candidates_remote_only.csv`
  - `ranked_candidates_remote_experimental_view.csv`
  - `top_remote_candidates.csv`
  - `top_remote_candidates.fasta`
- CV 中新增：`evaluation_view = remote_only`
- `remote_score` 调优应在远缘筛选层（identity/qcov/tcov gate）固定后进行，不建议同时频繁改 identity gate 和 ranking 权重。

最小 remote 配置示例：

```yaml
remote_discovery:
  enabled: true
  identity_max: 0.30
  qcov_min: 0.60
  tcov_min: 0.60
  require_missed_only: true
  export:
    top_n: 200
    make_fasta: true
    make_experimental_csv: true
```

兼容性：旧字段 `w_affinity / w_novelty / w_local_support` 仍可读取。

## 输出

### 保持兼容的主要输出

- `positive_clusters.csv`
- `positive_prototypes.npz`
- `positive_medoids.csv`
- `mmseqs_results.*`
- `hmm_results.*`
- `ranked_candidates.*`
- `cv_summary.csv`

### 新增输出

- `ranked_candidates_features.csv`
- `ranked_candidates_experimental_view.csv`
- `ranked_candidates_missed_only.csv`
- `ranked_candidates_easy_only.csv`
- `ranked_candidates_remote_only.csv`
- `ranked_candidates_remote_experimental_view.csv`
- `top_remote_candidates.csv`
- `top_remote_candidates.fasta`
- `remote_discovery_diagnostics.json`
- `remote_identity_bins.csv`
- `cv_fold_diagnostics.json`
- `cv_method_config_used.json`
- `feature_importance.csv`（learned scorer可用时）
- `ablation_summary.csv`

### merged 总榜 vs missed-only / remote-only 视图

- `ranked_candidates.*` / `rank`：保持与当前流程兼容的 merged 总榜。
- `ranked_candidates_missed_only.csv`：只看 missed 候选，避免 easy-hit 大量占位时把真正新候选压到几万名以后。
- `ranked_candidates_easy_only.csv`：只看 easy hits，便于单独复核传统近同源命中。
- `ranked_candidates_remote_only.csv`：只包含满足 remote gate 的候选，并按 `remote_score` 排序。
- `ranked_candidates_experimental_view.csv` 在 `remote_discovery.enabled=true` 时默认输出 remote-only 视角。

### easy-hit 的推荐理解

- easy 代表**高置信近邻集合**，不应等于“泛 MCO 同源集合”。
- 在 MCO-only 搜索空间中，`or + 宽阈值` 往往会让 easy_fraction 接近 1，这通常说明 easy 判定过宽，而不是 discovery 能力更强。
- 当前推荐默认值：
  - `decision_rule = consensus`
  - `profile = balanced`
  - `max_easy_fraction = 0.30`
- 当 easy_fraction 仍明显过高时，建议优先查看 `easy_hit_diagnostics.json`、`easy_hit_overlap_summary.csv` 和 missed-only 排名输出。

## 实验筛选建议

- discovery 阶段可使用 gold+silver 增强召回。
- 严格方法比较建议：`fixed clustering + leave_one_gold_family_out`。
- top-ranked 候选优先看：
  - `positive_support_score`
  - `reason_for_high_rank`
  - `confidence_tier`
  - `risk_flags`

## 运行

```bash
python run_pipeline.py
```

`config.yaml` 可直接改为 fixed+family-aware，或使用 `configs/examples/` 的示例。


## 主流程排序语义（已接线）

- 先做 `easy/missed` 分流（主流程显式使用上游 `easy_ids/missed_ids`，不在合并阶段隐式重分流）。
- `missed` 先构建特征，再按 `retrieval.scorer`（`heuristic` 或 `learned`）打分。
- 再按 `retrieval.easy_hit_policy` 合并：
  - `prepend`：easy 在前，missed 在后。
  - `merge`：按统一分数融合。
  - `interleave`：easy/missed 交错输出，减少 easy 全面占榜。
  - `prepend_with_cap`：easy 先放前面，但只对前部占位数量设上限。
- `ranked_candidates_experimental_view.csv` 即最终实验优先级视图；在 remote 模式下默认就是 remote-only 视图。
- 实验筛选默认优先参考 `experimental_priority_rank`（风险惩罚后的优先级），而不是仅看裸 `rank/final_score`。
- 当目标是远缘发现时，建议直接启用 `remote_discovery.enabled=true`，主实验输出会切换为 remote 榜单，而不是 easy/merged 总榜。
- easy 的默认判定不再是“任一工具命中即 easy”，而是 `consensus + balanced`：双工具支持优先，单工具命中只有在 strict 阈值下才进入 easy。

说明：
- `final_score` / `rank`：模型层综合排序（检索/打分结果）。
- `experimental_priority_score` / `experimental_priority_rank`：实验筛选层优先级（在 `final_score` 基础上做风险惩罚与轻量实验友好校正）。


## Experimental view 字段补充

`ranked_candidates_experimental_view.csv` 现在额外包含：
- `false_positive_risk`：综合风险分数（越高越需谨慎）
- `generic_mco_risk_score`：落在 generic MCO 密集背景区的风险
- `missed_rank` / `missed_experimental_priority_rank`：候选在 missed 子集内部的排序
- `rank_shift_due_to_easy`：由于 easy 占位导致的名次后移量
- `is_top_in_missed` / `is_top_in_merged`：是否进入各自视角的 top-N
- `why_hidden_by_easy`：当某候选在 missed 里很靠前但在 merged 里被 easy 遮蔽时给出原因说明
- `easy_confidence_class`：`both_support | mmseqs_strict_only | hmm_strict_only`
- `easy_reason`：easy 判定来源与使用的 rule/profile
- `easy_strength_score`：easy 证据强度摘要

实现说明（已落地到特征构建）：
- `generic_mco_risk_score` 由 `local_density_score`、`(1-positive_support_score)` 以及 MMseqs/HMM 支持弱信号组合得到，并与 `generic_mco_risk` flag 对齐校正。
- `false_positive_risk` 在 generic 风险基础上，再叠加 `novelty-support` 失配（novelty 虚高但 support/affinity 低）和弱 affinity 风险；同时受 `length_outlier/too_close_to_easy_hit` 等 flag 轻度增益。
- 两者均为 0~1，数值越高风险越大。
- 权重可通过 `retrieval.risk.*` 调整，默认参数偏保守（避免风险项反客为主）。

`experimental_priority_rank` 不再简单复制 `rank`：
- 先基于 `final_score`，再扣减 `false_positive_risk` 与 `generic_mco_risk_score`，用于实验优先级 triage。
- 惩罚强度可通过 `retrieval.experimental_priority.*` 调整。
- 默认还会对 `easy` 命中和 `high confidence_tier` 给出轻量 bonus（可配置），随后归一化得到 `experimental_priority_score`。

Top 导出行为：
- 常规模式下，`top_candidates_experimental.csv` 与 `top_candidates.fasta` 按 `experimental_priority_rank` 前 N 导出。
- remote 模式下，这两个文件默认改为从 **remote 榜单** 导出；同时还会额外输出 `top_remote_candidates.csv` 与 `top_remote_candidates.fasta`。

CV / 消融输出：
- `cv_summary.csv` 与 `ablation_summary.csv` 现在包含 `evaluation_view`：
  - `overall_merged`
  - `missed_only`
  - `remote_only`
- `missed_only` 指标只在 missed 子集内部计算，更适合回答“新的远同源候选是否被排到前面”。
- `remote_only` 指标只在满足 remote gate 的子空间内部计算；若某个 fold 的 holdout positives 不落入 remote 空间，会明确记录为该视图无可评估正例（NaN），不会静默忽略。

easy-hit 诊断输出：
- `easy_hit_diagnostics.json`：记录 easy/missed 总量、easy_fraction、单工具/双工具 easy 计数、是否超过上限等。
- `easy_hit_overlap_summary.csv`：记录全局及按推断 family 的 easy_fraction 摘要。
- `easy_hit_thresholds_used.json`：记录最终实际使用的 easy 阈值与 profile，便于复现。

## Learned 分支使用建议（本轮收紧）

- discovery 与 CV 的 learned 训练数据都通过同一个 `build_learned_training_data(...)` 构建。
- CV 的 learned 最终排序也走与 discovery 一致的 `rank_candidates_with_policy(...)` merge 语义（`prepend/merge` 两种 policy 一致处理）。
- 正样本训练特征采用 leave-one-out 方式构建，避免把自身当作锚点造成过于乐观的支持信号。
- 默认仍推荐 `heuristic` 作为稳定主线；`learned` 作为可选增强，训练失败或样本不足时自动回退 heuristic。

## 当前局限（诚实说明）

- 风险量化仍是规则型可解释模型，不是监督风险校准器；建议与 `risk_flags`、邻近正样本信息联合判读。
- learned 分支当前为轻量 logistic + 加权采样/加权训练，目标是提升 top-K 实验筛选实用性，不保证在所有数据上优于 heuristic。

建议优先挑选：`high confidence_tier` 且 `positive_support_score` 高、`false_positive_risk` 低的候选。
