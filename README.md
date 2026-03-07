# MnOx MCO 远同源候选发现工程

本项目是一个**本地可运行**的 Python 工程（非 CLI 子命令风格），用于在 UniRef 注释为 MCO 的大规模未标注序列中，检索可能具有 Mn(II) 氧化活性的远同源候选。

## 目录结构

```text
.
├── config.yaml
├── run_pipeline.py
├── scripts/
│   └── setup_centos.sh
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

## CentOS 使用说明（推荐）

### 方案 A：一键脚本（推荐）

```bash
bash scripts/setup_centos.sh
conda activate mnox
python run_pipeline.py
```

> 说明：脚本会尝试用 `dnf/yum` 安装 `hmmer/mafft/muscle`，并用 conda 安装 `mmseqs2` 与 Python 依赖。

### 方案 B：手动安装

1. 系统依赖（CentOS/RHEL/Rocky/AlmaLinux）

```bash
# dnf 系列（CentOS Stream/Rocky/Alma）
sudo dnf install -y epel-release
sudo dnf install -y hmmer mafft muscle git wget

# CentOS 7 可用 yum
sudo yum install -y epel-release
sudo yum install -y hmmer mafft muscle git wget
```

2. Conda 环境与 Python 包（**强烈建议重包都用 conda**）

```bash
conda create -n mnox python=3.10 -y
conda activate mnox

# 生信工具 + 科学计算栈（避免 pip 在老系统上源码编译失败）
conda install -c conda-forge -c bioconda -y \
  mmseqs2 hmmer mafft muscle \
  numpy pandas scipy scikit-learn matplotlib pyarrow biopython pyyaml transformers

# PyTorch（CPU 版）
conda install -c pytorch -y pytorch cpuonly
```


3. 准备数据并修改 `config.yaml`

- `positives.fasta`
- `unlabeled_mco.fasta`

4. 运行

```bash
python run_pipeline.py
```

输出在 `runs/YYYYMMDD_HHMMSS/`。

## 运行与性能建议（CentOS）

- `mmseqs_tmp` 建议放在本地 SSD（可在 `config.yaml` 的 `mmseqs.tmp_dir` 调整）。
- 大规模 embedding 建议优先 GPU；CPU 跑时可适当减小 `esm.batch_size`。
- 对 16 万序列建议保留 `esm.shard_size=5000~20000`，并确保磁盘空间充足。

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


## 常见报错与修复（CentOS）

### 1) `pip install pandas` 触发源码编译失败（`stdatomic.h` / gcc 4.8.5）

症状：你日志中的 `fatal error: stdatomic.h: No such file or directory`、`__has_builtin not detected`。  
原因：CentOS 7 默认编译器太老，pip 没拿到 wheel 后转源码编译 pandas 失败。

**推荐修复（不要用 pip 编 pandas）**：

```bash
conda activate mnox
conda install -c conda-forge -y numpy pandas scipy scikit-learn matplotlib pyarrow biopython pyyaml transformers
conda install -c pytorch -y pytorch cpuonly
```

### 2) 如果你必须用 pip（不推荐）

先升级编译工具链（如 devtoolset/gcc>=9）和 Python 头文件，再安装；但这条路径复杂、易踩坑。对本项目建议始终优先 conda 二进制包。
