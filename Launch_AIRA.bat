@echo off
title Launch AIRA JARVIS
echo ===================================================
echo           Starting AIRA JARVIS Assistant
echo ===================================================

echo [1/3] Preparing Ollama AI Engine...
netstat -ano | findstr ":11434" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    taskkill /F /IM "ollama app.exe" >nul 2>&1
    set OLLAMA_ORIGINS=*
    start "" /B "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
)

echo [2/3] Starting AIRA Server on http://localhost:8000...
cd /d "%~dp0AIRA"
if not exist "server.py" cd /d "%~dp0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
start "AIRA Backend Server" /min python server.py

echo [3/3] Waiting for AI services to be ready...
powershell -Command "for ($i=0; $i -lt 15; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/status' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { break } } catch {}; Start-Sleep -Milliseconds 600 }"

echo.
echo AIRA is ready! Opening interface...
start "" http://localhost:8000
