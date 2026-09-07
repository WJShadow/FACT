@echo off
setlocal
set "SCRIPT=%~dp0Download_FACT_GUI.ps1"

if not exist "%SCRIPT%" (
    echo Missing downloader: "%SCRIPT%"
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
set "RESULT=%ERRORLEVEL%"

if not defined FACT_NO_PAUSE pause
exit /b %RESULT%
