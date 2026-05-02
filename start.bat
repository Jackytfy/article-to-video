@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ===================================
echo Article to Video - 一键启动
echo ===================================

REM 检查端口是否被占用
netstat -ano | find "LISTENING" | find "8000" > nul
if %errorlevel% == 0 (
    echo API 服务已在运行 (端口 8000)
) else (
    echo 正在启动 API 服务...
    start "API Server" cmd /c "uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
    echo 等待服务启动...
    timeout /t 5 /nobreak > nul
)

echo 正在打开前端页面...
start http://127.0.0.1:8000

echo.
echo 服务已启动！
echo 前端地址: http://127.0.0.1:8000
echo 按任意键关闭此窗口...
pause > nul
