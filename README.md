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

- `retrieval.scorer`: `heuristic | learned`
- `retrieval.easy_hit_policy`: `prepend | merge`
- `retrieval.support_topk`
- `retrieval.density_knn_k`
- `retrieval.heuristic.*`
- `retrieval.learned.*`

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

- 先做 `easy/missed` 分流。
- `missed` 先构建特征，再按 `retrieval.scorer`（`heuristic` 或 `learned`）打分。
- 再按 `retrieval.easy_hit_policy` 合并：
  - `prepend`：easy 在前，missed 在后。
  - `merge`：按统一分数融合。
- `ranked_candidates_experimental_view.csv` 即最终实验优先级视图。
