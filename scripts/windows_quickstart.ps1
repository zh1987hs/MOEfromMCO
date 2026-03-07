param(
  [string]$ProjectRoot = "D:\MOE",
  [string]$PythonExe = "py",
  [string]$ConfigPath = "config/default.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[1/6] 创建虚拟环境"
& $PythonExe -3.10 -m venv .venv

Write-Host "[2/6] 安装依赖"
& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\python -m pip install -r requirements.txt
& .\.venv\Scripts\python -m pip install -e .

Write-Host "[3/6] 检测外部工具"
& .\.venv\Scripts\python -m mnox_retrieval.cli --config $ConfigPath doctor

Write-Host "[4/6] 生成模拟数据"
& .\.venv\Scripts\python -m mnox_retrieval.cli --config $ConfigPath simulate-data --out-dir outputs/sim_data

Write-Host "[5/6] 跑完整 demo（当前未安装 BLAST/HMMER 时会自动回退）"
& .\.venv\Scripts\python -m mnox_retrieval.cli --config $ConfigPath run-demo --sim-dir outputs/sim_data --out-dir outputs/demo

Write-Host "[6/6] 输出文件"
Write-Host "- outputs/demo/ranked_candidates.csv"
Write-Host "- outputs/demo/top_candidates.fasta"
Write-Host "- outputs/demo/evaluation_summary.json"
Write-Host "- outputs/demo/tool_detection.json"

Write-Host "完成。若后续要启用真实 BLAST/HMMER，请先执行："
Write-Host ".\\.venv\\Scripts\\python -m mnox_retrieval.cli install-tools --method conda"
