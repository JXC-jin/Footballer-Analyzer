# ============================================================
# 建立本地 CPU 基准测试环境（无 CUDA 的机器）
#   用法:  & .\scripts\setup_bench_env.ps1
#   产物:  .venv\  （已由 .gitignore 排除）
# ============================================================
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Output "[venv] 创建 $venv"
    python -m venv $venv
} else {
    Write-Output "[venv] 复用已有 $venv"
}

Write-Output "[pip] 升级 pip"
& $py -m pip install --upgrade pip

Write-Output "[pip] 安装 PyTorch CPU 版（约 200MB，走 pytorch 官方 cpu 源）"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Output "[pip] 安装 ultralytics / opencv / onnxruntime-directml"
& $py -m pip install ultralytics opencv-python onnxruntime-directml onnx

Write-Output "[verify]"
& $py -c "import torch, cv2, ultralytics, onnxruntime as ort; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__); print('ort', ort.__version__, ort.get_available_providers())"
Write-Output "[done]"
