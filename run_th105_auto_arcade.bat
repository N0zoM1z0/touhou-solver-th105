@echo off
setlocal

if not defined TH105_GAME_DIR set "TH105_GAME_DIR=D:\Entertainment\Game\Touhou\[th105] 东方绯想天 (汉化版+日文版)"
if not defined TH105_PYTHON set "TH105_PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"

if not exist "%TH105_PYTHON%" (
  echo Missing Windows Python: "%TH105_PYTHON%"
  exit /b 1
)

"%TH105_PYTHON%" "%~dp0scripts\run_th105_agent.py" auto-arcade --launch --p1-character sakuya --battle-seconds 300 %*
exit /b %errorlevel%
