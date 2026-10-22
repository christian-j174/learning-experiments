# Crea (o actualiza) el entorno .venv para los notebooks NOAA en Windows.
# Uso, desde la carpeta learning-experiments en PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. uv: gestor rápido de Python y paquetes. Se instala con el Python que ya tengas.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    py -m pip install --user --quiet uv
    $uv = "py -m uv"
} else {
    $uv = "uv"
}

# 2. Python 3.11 (TensorFlow 2.15 no soporta 3.12+) y el entorno virtual.
Invoke-Expression "$uv python install 3.11"
if (-not (Test-Path .venv)) {
    Invoke-Expression "$uv venv .venv --python 3.11"
}

# 3. Dependencias de todos los proyectos.
Invoke-Expression "$uv pip install --python .venv\Scripts\python.exe -r requirements.txt"

# 4. Kernel de Jupyter con nombre fijo, para elegirlo en JupyterLab o VS Code.
.venv\Scripts\python.exe -m ipykernel install --user --name noaa-ml --display-name "Python (NOAA ML)"

Write-Host ""
Write-Host "Listo. Para abrir JupyterLab:"
Write-Host "  .venv\Scripts\activate"
Write-Host "  jupyter lab"
