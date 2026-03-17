# MnOx MCO candidate prioritization pipeline

本项目定位是：
**remote-homology-guided candidate prioritization pipeline for putative Mn(II)-oxidizing MCO discovery**。

> 重点是提升 top-K 候选（前10/20/50）实验筛选价值，而不是直接输出 definitive functional annotation。

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
- `retrieval.easy_hit_policy`: `prepend | merge`
- `retrieval.support_topk`
- `retrieval.density_knn_k`
- `retrieval.heuristic.*`
- `retrieval.learned.*`
- `retrieval.risk.*`：风险量化权重
- `retrieval.experimental_priority.*`：实验优先级排名中的风险惩罚权重

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
- `cv_fold_diagnostics.json`
- `cv_method_config_used.json`
- `feature_importance.csv`（learned scorer可用时）
- `ablation_summary.csv`

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
- `ranked_candidates_experimental_view.csv` 即最终实验优先级视图。
- 实验筛选默认优先参考 `experimental_priority_rank`（风险惩罚后的优先级），而不是仅看裸 `rank/final_score`。

说明：
- `final_score` / `rank`：模型层综合排序（检索/打分结果）。
- `experimental_priority_score` / `experimental_priority_rank`：实验筛选层优先级（在 `final_score` 基础上做风险惩罚与轻量实验友好校正）。


## Experimental view 字段补充

`ranked_candidates_experimental_view.csv` 现在额外包含：
- `false_positive_risk`：综合风险分数（越高越需谨慎）
- `generic_mco_risk_score`：落在 generic MCO 密集背景区的风险

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
- `top_candidates_experimental.csv` 与 `top_candidates.fasta` 都按 `experimental_priority_rank` 前 N 导出（不再按 DataFrame 原顺序）。

## Learned 分支使用建议（本轮收紧）

- discovery 与 CV 的 learned 训练数据都通过同一个 `build_learned_training_data(...)` 构建。
- CV 的 learned 最终排序也走与 discovery 一致的 `rank_candidates_with_policy(...)` merge 语义（`prepend/merge` 两种 policy 一致处理）。
- 正样本训练特征采用 leave-one-out 方式构建，避免把自身当作锚点造成过于乐观的支持信号。
- 默认仍推荐 `heuristic` 作为稳定主线；`learned` 作为可选增强，训练失败或样本不足时自动回退 heuristic。

## 当前局限（诚实说明）

- 风险量化仍是规则型可解释模型，不是监督风险校准器；建议与 `risk_flags`、邻近正样本信息联合判读。
- learned 分支当前为轻量 logistic + 加权采样/加权训练，目标是提升 top-K 实验筛选实用性，不保证在所有数据上优于 heuristic。

建议优先挑选：`high confidence_tier` 且 `positive_support_score` 高、`false_positive_risk` 低的候选。
