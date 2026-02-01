$ErrorActionPreference = 'Stop'

# Starts FastAPI server that also serves /web and /data.
# Open:
# - http://localhost:8000/web/
# - http://localhost:8000/api/health

$python = "F:/202507/AIH_SURVEY/.venv/Scripts/python.exe"

Write-Host "Starting API at http://localhost:8000" -ForegroundColor Cyan
& $python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
