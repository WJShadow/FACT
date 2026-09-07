@echo off
setlocal
title Install FACT fact_py311 environment (online)

set "INSTALLER=%~dp0Installation\install_fact.ps1"
if not exist "%INSTALLER%" (
    echo ERROR: FACT installer core was not found:
    echo        %INSTALLER%
    set "RESULT=1"
    goto :finish
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%" -Mode Online
set "RESULT=%ERRORLEVEL%"

:finish
if not defined FACT_NO_PAUSE pause
endlocal & exit /b %RESULT%
