# MnOx MCO 远同源候选发现工程

本项目用于从大规模未标注 MCO 序列中发现潜在 Mn(II)-oxidizing MCO 候选。保留原有主流程（QC→embedding→聚类/原型→MMseqs2/HMMER→排序→CV），并新增 **gold/silver positives** 与 **fixed cluster** 公平比较框架。

## 关键新增能力（最小改动，兼容旧流程）

1. 显式支持正样本元数据（gold/silver/family）
2. 支持 `positive_clustering.mode = fixed | auto | none`
3. fixed 模式下：**簇成员固定**，但 prototype/medoid 仍按当前模型 embedding 重算
4. 新增 `evaluation.mode = leave_one_gold_family_out`，并支持 `leakage_guard`
5. 旧配置仍可用：`auto + loco` 兼容保留

---

## 正样本元数据格式

`positive_metadata.csv`：

```csv
positive_id,tier,gold_family,seed_gold_id
WP_xxx,gold,FAM01,WP_xxx
UniRef90_xxx,silver,FAM01,WP_xxx
```

- `tier`：`gold` 或 `silver`
- `gold_family`：评估与泄漏防护的关键分组
- `seed_gold_id`：silver 来源追踪

固定簇文件 `positive_clusters_fixed.csv`：

```csv
positive_id,cluster
WP_xxx,0
UniRef90_xxx,0
```

---

## clustering 三种模式

- `fixed`：适合 35M vs 150M 的**公平比较**（簇成员不随模型变化）
- `auto`：按当前 embedding 自动聚类（用于探索分析）
- `none`：全部正样本视为一个簇（小样本场景）

为什么 fixed 更公平：
- 自动聚类会随 embedding 模型变化，比较会混入“簇定义变化”因素；
- fixed 模式将簇定义固定，仅比较“当前模型表示能力”。

---

## 评估模式

- `loco`：原有 leave-one-cluster-out
- `leave_one_gold_family_out`：新增 family-aware 评估
  - 每折留出一个 `gold_family`
  - 若 `leakage_guard=true`，该 family 的 gold+silver 都从训练锚点中排除
  - 输出 `MRR / recall@K / ef@K`

---

## 配置示例

- fixed + family-aware：`configs/examples/config_fixed_gold_family.yaml`
- auto + LOCO：`configs/examples/config_auto_loco.yaml`

默认配置：`config.yaml`。

---

## 运行

```bash
python run_pipeline.py
```

输出目录：`runs/YYYYMMDD_HHMMSS/`

---

## 输出兼容性说明

### 保持不变

- `positive_clusters.csv`
- `positive_prototypes.npz`
- `positive_medoids.csv`
- `mmseqs_results.parquet/csv`
- `hmm_results.parquet/csv`
- `ranked_candidates.parquet/csv`
- `cv_summary.csv`

### 新增/增强

- `positive_metadata_used.csv`（本次运行实际使用的正样本元数据）
- `positive_clusters.csv` 新增来源信息（`cluster_source`）及 metadata 合并列
- `cv_summary.csv` 增加 `evaluation_mode`，family-aware 时包含 `fold_gold_family` 和 `leakage_guard`

---

## 发现与评估的建议

- discovery 阶段可使用 expanded positives（gold+silver）增强召回。
- 严格方法比较和解释建议使用 `fixed clustering + leave_one_gold_family_out`，避免 family leakage，并保证模型间公平性。
