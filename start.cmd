@echo off
echo ================================
echo  智能面试模拟与反馈系统
echo ================================
echo.

set "ROOT=%~dp0"

echo 正在安装依赖...
cd /d "%ROOT%"
.venv\Scripts\python.exe -c "pass" 2>nul
if errorlevel 1 (
    echo 创建虚拟环境...
    uv venv
    .venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>nul
    uv pip install -r requirements.txt
)

echo.
echo ================================
echo  启动面试模拟
echo ================================
echo.

set PYTHONPATH=%ROOT%
.venv\Scripts\python.exe src/main.py %*

echo.
pause
