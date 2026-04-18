@echo off
setlocal
cd /d "%~dp0"

set VENV_PYTHON=%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe
set ROUTELLM_URL=http://localhost:6060/v1/models

echo ============================================
echo  RouteLLM 啟動中...
echo ============================================

:: Check if already running
curl -s --max-time 2 %ROUTELLM_URL% >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] RouteLLM 已在運行，跳過啟動
    goto :test
)

:: Start RouteLLM in a new window
start "RouteLLM" "%VENV_PYTHON%" start.py

:: Wait up to 30 seconds for RouteLLM to be ready
echo 等待 RouteLLM 就緒...
set /a attempts=0
:wait_loop
    timeout /t 2 /nobreak >nul
    curl -s --max-time 2 %ROUTELLM_URL% >nul 2>&1
    if %errorlevel% == 0 goto :ready
    set /a attempts+=1
    echo   [%attempts%] 還在啟動中...
    if %attempts% lss 15 goto :wait_loop

echo [ERROR] RouteLLM 啟動逾時，請檢查 RouteLLM 視窗的錯誤訊息
pause
exit /b 1

:ready
echo [OK] RouteLLM 已就緒！

:test
echo.
echo ============================================
echo  執行路由測試...
echo ============================================
"%VENV_PYTHON%" test_routing.py

echo.
echo ============================================
echo  完成！按任意鍵關閉
echo ============================================
pause
