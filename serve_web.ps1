$ErrorActionPreference = 'Stop'

# Run a simple static server at workspace root so /data and /web are accessible.
# Then open: http://localhost:8000/web/

$python = "F:/202507/AIH_SURVEY/.venv/Scripts/python.exe"

Write-Host "Serving http://localhost:8000/web/" -ForegroundColor Cyan
& $python -m http.server 8000
