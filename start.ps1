$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Get-Command py -ErrorAction SilentlyContinue
if (-not (Test-Path ".venv")) {
    if ($python) { py -3.11 -m venv .venv }
    else { python -m venv .venv }
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "Python 3.11 non détecté. Installe Python depuis https://www.python.org/downloads/ puis relance." }
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
& $venvPython -m streamlit run app.py --server.port 8501
