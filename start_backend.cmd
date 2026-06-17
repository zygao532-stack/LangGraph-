@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo Starting interview backend on http://127.0.0.1:8001 ...
.venv\Scripts\python.exe -c "import uvicorn; uvicorn.run('backend.app.main:app', host='127.0.0.1', port=8001, reload=True)"
pause
