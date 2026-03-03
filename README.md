# MnOx Retrieval (MCO 远同源候选发现)

本项目实现一个**正样本驱动（positive-unlabeled / one-class retrieval）**的端到端管线，用于在多铜氧化酶（MCO）序列集合中检索潜在锰氧化活性远同源候选。默认可离线运行（模拟数据 + fallback embedding），并预留真实 FASTA / ESM / BLAST / HMMER 接口。

## 方法概览
- 先在正样本 embedding 空间建模正类流形（多 prototype）。
- 先算 BLAST-like / HMM-like baseline 分数。
- 对 easy-hit（易被传统方法命中）施加 penalty。
- 在 hard/missed 区域优先排序 embedding 上接近正样本、且 novelty 高的候选。
- 提供 leave-one-cluster-out 留簇验证。

## 模拟数据设计
- 长度默认 450–700 aa。
- 融入 MCO-like motif + 酸性片段偏好（D/E enrichment）。
- 未标注池包含：normal MCO-like、easy-hit 近同源、远同源 hidden positives（spike-in）。
- hidden positives 在序列 identity 上更低，但在 embedding 上保持可检索弱规律。

## 安装
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
```

## CLI
```bash
python -m mnox_retrieval.cli --help
python -m mnox_retrieval.cli simulate-data --out-dir outputs/sim_data
python -m mnox_retrieval.cli run-demo --sim-dir outputs/sim_data --out-dir outputs/demo
python -m mnox_retrieval.cli cross-validate --sim-dir outputs/sim_data --out-dir outputs/cv
python -m mnox_retrieval.cli rank-real --positive-fasta path/to/positives.fasta --unlabeled-fasta path/to/uniref_mco.fasta --out-dir outputs/real
python -m mnox_retrieval.cli export-top --ranking-csv outputs/demo/ranked_candidates.csv --source-fasta outputs/sim_data/unlabeled.fasta --top-n 100 --out-fasta outputs/demo/top_candidates.fasta
python -m mnox_retrieval.cli plot-report --cv-metrics outputs/cv/cv_metrics_by_fold.csv --out-dir outputs/plots
```

## 主要输出
- `ranked_candidates.csv`
- `top_candidates.fasta`
- `evaluation_summary.json`
- `config_used.yaml`
- `rank_distribution.png`
- `embedding_pca.png`
- 留簇验证：`cv_metrics_by_fold.csv`, `cv_summary.csv`, `cv_recall_curve.png`

## 可替换真实组件
- BLAST-like: 目前为纯 Python 近似；可在同接口替换为 BLAST+ `blastp`。
- HMM-like: 目前为简化 profile；可替换为 HMMER `hmmbuild/hmmsearch`。
- Embedding: 默认 fallback；若配置 ESM 模型路径并安装 `torch/transformers`，可切换真实 ESM。

## 局限性
- 模拟数据不代表真实进化与结构生物学复杂性。
- baseline 为近似实现，主要用于离线 demo 与流程验证。
- 真正应用到 UniRef 大规模检索时，建议接入真实 BLAST/HMMER + GPU ESM。

## 附加脚本
```bash
python scripts/demo_analysis.py
```
自动执行 demo、CV 并生成 markdown 报告。
