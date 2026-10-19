@echo off
setlocal
cd /d "%~dp0"

echo Checking Python dependencies...
py -c "import numpy, pandas, matplotlib, yfinance" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages. This needs an internet connection...
    py -m pip install -r requirements_ma_distance.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Check Python and internet access.
        pause
        exit /b 1
    )
)

echo.
echo Running offline self-test...
py ma_distance_research.py --self-test
if errorlevel 1 (
    echo Self-test failed. The backtest will not be started.
    pause
    exit /b 1
)

echo.
echo Starting Yahoo Finance research backtest...
py ma_distance_research.py %*
if errorlevel 1 echo The backtest ended with an error. Read the messages above.
pause
