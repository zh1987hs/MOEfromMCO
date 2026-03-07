# MnOx Retrieval (MCO 远同源候选发现)

本项目实现一个**正样本驱动（positive-unlabeled / one-class retrieval）**的端到端管线，用于在多铜氧化酶（MCO）序列集合中检索潜在锰氧化活性远同源候选。

## 你现在可以“一步到位”做真实检索
- `external.blast_mode=auto`：检测到 BLAST+（`blastp/makeblastdb`）就走真实 BLAST，缺失时自动回退 BLAST-like
- `external.hmm_mode=auto`：检测到 HMMER（`phmmer`）就走真实 HMMER，缺失时自动回退 HMM-like
- `embedding.mode=auto`：检测到可用 ESM 权重就走真实 ESM，否则自动回退离线 embedding

> 运行后会输出 `tool_detection.json`，明确本次是否命中了真实后端。

## 安装（Windows 友好）
推荐不激活虚拟环境，直接用 venv 内 python：
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

PowerShell 若报执行策略问题：
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 外部工具安装（BLAST+/HMMER）
先看当前环境检测：
```bash
python -m mnox_retrieval.cli doctor
```

打印安装建议命令：
```bash
python -m mnox_retrieval.cli install-tools --method conda
python -m mnox_retrieval.cli install-tools --method choco
```

## 国内离线 ESM 建议
1. 在可联网机器下载 ESM 模型目录（transformers 格式）
2. 拷贝到本地，例如 `D:\models\esm2_t33_650M_UR50D`
3. 配置 `config/default.yaml`：
```yaml
embedding:
  mode: esm
  esm_model_path: D:/models/esm2_t33_650M_UR50D
  pooling: mean
```

## CLI
```bash
python -m mnox_retrieval.cli --help
python -m mnox_retrieval.cli simulate-data --out-dir outputs/sim_data
python -m mnox_retrieval.cli run-demo --sim-dir outputs/sim_data --out-dir outputs/demo
python -m mnox_retrieval.cli cross-validate --sim-dir outputs/sim_data --out-dir outputs/cv
python -m mnox_retrieval.cli rank-real --positive-fasta path/to/positives.fasta --unlabeled-fasta path/to/uniref_mco.fasta --out-dir outputs/real
# 注意: path/to/... 是占位路径，需替换为真实存在的 FASTA 文件
python -m mnox_retrieval.cli export-top --ranking-csv outputs/real/ranked_candidates.csv --source-fasta path/to/uniref_mco.fasta --top-n 200 --out-fasta outputs/real/top200.fasta
python -m mnox_retrieval.cli plot-report --cv-metrics outputs/cv/cv_metrics_by_fold.csv --out-dir outputs/plots
```

## 输出文件
- `ranked_candidates.csv`
- `top_candidates.fasta`
- `evaluation_summary.json`
- `config_used.yaml`
- `tool_detection.json`
- `rank_distribution.png`
- `embedding_pca.png`

## 方法概览
- 正样本 embedding 空间建模（多 prototype）
- BLAST/HMM baseline 先打分，easy-hit 降权
- 在 hard/missed 区域优先找 embedding 接近且 novelty 高的候选
- 留一簇验证（leave-one-cluster-out）

## 局限性
- 模拟数据不等价真实生物进化复杂性
- 默认 HMMER 真实后端使用 `phmmer`（无需先建 profile）；若你后续要更严格 profile-HMM，可按同接口接 `hmmbuild+hmmsearch`


## D:\MOE 本地一步一步出结果（未安装 BLAST/HMMER 也可先跑）
下面按你当前状态（外部工具未安装）给出最稳流程：

### Step 0: 打开终端并进入项目目录
```powershell
cd D:\MOE
```

### Step 1: 创建并使用虚拟环境（推荐不激活）
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

### Step 2: 看当前是否检测到外部工具
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli doctor
```
- 若显示 `MISSING`，不用担心：当前会自动走 fallback，依然能出结果。

### Step 3: 先跑模拟数据，确认全链路
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli simulate-data --out-dir outputs/sim_data
.\.venv\Scripts\python -m mnox_retrieval.cli run-demo --sim-dir outputs/sim_data --out-dir outputs/demo
```

### Step 4: 查看结果文件
- `outputs/demo/ranked_candidates.csv`（主排序结果）
- `outputs/demo/top_candidates.fasta`（Top 候选）
- `outputs/demo/evaluation_summary.json`（指标）
- `outputs/demo/tool_detection.json`（本次是否命中真实工具）

### Step 5: 跑真实 FASTA（你自己的正样本+未标记集合）
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli rank-real --positive-fasta D:/MOE/data/positives.fasta --unlabeled-fasta D:/MOE/data/uniref_mco.fasta --out-dir outputs/real
```
完成后看：
- `outputs/real/ranked_candidates.csv`
- `outputs/real/top_candidates.fasta`
- `outputs/real/tool_detection.json`

### Step 6（可选）: 再升级到真实 BLAST/HMMER
先查看建议命令：
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli install-tools --method conda
```
安装后再执行：
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli doctor
```
如果显示 `blastp/makeblastdb/phmmer: OK`，后续 `rank-real` 会自动优先走真实后端（配置是 `auto`）。

### 一键脚本（可选）
项目里已提供：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows_quickstart.ps1 -ProjectRoot D:\MOE
```


## 常见报错：`No module named mnox_retrieval`
如果你在 `D:\MOE` 执行：
```powershell
python -m mnox_retrieval.cli doctor
```
出现该错误，通常是解释器路径或安装方式问题。

### 快速自检（按顺序）
1. 确认在项目根目录：
```powershell
cd D:\MOE
```
2. 看当前 Python 是不是你期望的：
```powershell
python -c "import sys; print(sys.executable)"
```
3. 推荐直接用 venv 的 python（避免路径串环境）：
```powershell
.\.venv\Scripts\python -m mnox_retrieval.cli doctor
```

### 现在这个仓库已支持“源码直跑”
我已加入桥接包，所以在仓库根目录下可直接执行：
```powershell
python -m mnox_retrieval.cli doctor
```
即使你还没先 `pip install -e .`，也可以找到 `src/mnox_retrieval`。

### 仍建议的标准做法
```powershell
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m mnox_retrieval.cli doctor
```
