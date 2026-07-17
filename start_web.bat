@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Prefer: python -m pip ... (avoids pip-shim GBK issues on Chinese Windows)
set HOST=127.0.0.1
set PORT=8765

echo ============================================================
echo   AlphaMaster Web Console
echo   http://%HOST%:%PORT%
echo ============================================================
echo.

echo [1/3] Releasing port %PORT% if occupied ...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  echo   taskkill PID %%P
  taskkill /F /PID %%P >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo [2/3] Starting uvicorn ...
start "AlphaMaster Server" python run_web.py --host %HOST% --port %PORT%

echo [3/3] Warming up APIs (torch/fastapi first import can take ~6s) ...
set /a TRIES=0
:wait_ready
set /a TRIES+=1
if %TRIES% GTR 90 goto ready_fail
timeout /t 1 /nobreak >nul
python -c "import urllib.request; urllib.request.urlopen('http://%HOST%:%PORT%/api/overview', timeout=2).read(); urllib.request.urlopen('http://%HOST%:%PORT%/api/ai/providers', timeout=2).read(); urllib.request.urlopen('http://%HOST%:%PORT%/api/strategies', timeout=2).read()" 2>nul
if errorlevel 1 goto wait_ready

echo.
echo Ready. Opening browser ...
start "" "http://%HOST%:%PORT%/"
echo.
echo Server window title: AlphaMaster Server
echo Press any key to close this launcher (server keeps running)...
pause >nul
exit /b 0

:ready_fail
echo.
echo ERROR: server did not become ready within ~90s.
echo Check the "AlphaMaster Server" window for import/dependency errors.
echo Tip: python -m pip install -r requirements.txt
echo.
pause
exit /b 1
