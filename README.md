# MnOx MCO 远同源候选发现工程

本项目是一个**本地可运行**的 Python 工程（非 CLI 子命令风格），用于在 UniRef 注释为 MCO 的大规模未标注序列中，检索可能具有 Mn(II) 氧化活性的远同源候选。

## 目录结构

```text
.
├── config.yaml
├── run_pipeline.py
├── mnox/
│   ├── __init__.py
│   ├── cluster.py
│   ├── esm_embed.py
│   ├── eval_cv.py
│   ├── hmmer.py
│   ├── io_fasta.py
│   ├── mmseqs.py
│   ├── plots.py
│   ├── qc.py
│   ├── retrieval.py
│   └── utils.py
└── README.md
```

## 本地最简运行步骤

1. 创建环境并安装 Python 依赖

```bash
conda create -n mnox python=3.10 -y
conda activate mnox
pip install biopython numpy pandas scikit-learn scipy pyyaml matplotlib torch transformers pyarrow
```

2. 安装外部工具（MMseqs2 / HMMER / MAFFT 或 MUSCLE）

```bash
# 推荐
conda install -c bioconda mmseqs2 hmmer mafft muscle -y

# macOS 可选
brew install mmseqs2 hmmer mafft muscle
```

3. 准备数据文件并修改 `config.yaml`

- `positives.fasta`
- `unlabeled_mco.fasta`

4. 运行

```bash
python run_pipeline.py
```

输出在 `runs/YYYYMMDD_HHMMSS/`。

### Windows（PowerShell）建议

```powershell
# 1) 建议先安装 Miniconda，再创建环境
conda create -n mnox python=3.10 -y
conda activate mnox
pip install biopython numpy pandas scikit-learn scipy pyyaml matplotlib torch transformers pyarrow

# 2) 外部工具优先用 bioconda（推荐）
conda install -c bioconda mmseqs2 hmmer mafft muscle -y

# 3) 如果 mmseqs2 在原生 Windows 不稳定，建议用 WSL2 运行同一项目目录
```


## 结果说明

主要输出包括：

- `qc_summary.json`, `kept_ids_*.txt`
- `embeddings/*`（分片缓存 + 断点）
- `positive_clusters.csv`, `positive_prototypes.npz`, `positive_medoids.csv`
- `mmseqs_results.parquet/csv`, `hmm_results.parquet/csv`
- `easy_ids.txt`, `missed_ids.txt`
- `ranked_candidates.parquet/csv`
- `top_candidates.fasta`, `top_candidates_by_cluster/*`
- `cv_summary.csv`, `plots/recall_at_k.png`, `plots/mrr_box.png`

## 为什么这是正样本驱动检索，而不是二分类

因为这里没有可靠负样本。`unlabeled_mco.fasta` 不能当“负类”，否则会把潜在阳性误标。流程以高置信正样本为锚点，通过序列检索 + embedding 相似性 + 局部支持度进行排序，属于 positive-unlabeled（PU）检索范式。

## missed_set 的意义

`missed_set` = MMseqs2 与 HMMER 都不易命中的序列子集，是传统同源检索难覆盖的区域。对该集合再做 embedding 原型检索与组合打分，可以聚焦“可能远同源”的高价值候选。

## LOCO 验证如何评估远同源可找回性

留一簇（LOCO）把某个正样本簇整体隐藏成“未知家族”，训练/检索只使用其余簇。若方法能在混入背景中把隐藏簇排到前列，则说明对远同源家族的可找回性更强。输出 Recall@K、MRR、EF@K 对比 MMseqs2-only / HMMER-only / embedding-only / hybrid。

## 局限性

- 排序结果仍是计算候选，最终需要实验验证。
- embedding 可能捕捉谱系/长度等非功能信号。
- MSA/HMM 质量受簇内多样性与比对质量影响。
- 超大规模计算对 GPU/CPU/存储要求较高。
