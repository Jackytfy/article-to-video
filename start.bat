@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================
echo Article to Video - Startup
echo ===================================
echo.

REM Check if port 8000 is in use
netstat -ano | find "LISTENING" | find "8000" >nul
if %errorlevel%==0 (
    echo API service is already running on port 8000
    echo Please open: http://127.0.0.1:8000
) else (
    echo Starting API service...
    start "A2V Server" cmd /k "uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
    timeout /t 3 /nobreak >nul
    echo Service started!
)
echo.
echo Open in browser: http://127.0.0.1:8000
echo.
echo Press any key to exit this window...
pause >nul
