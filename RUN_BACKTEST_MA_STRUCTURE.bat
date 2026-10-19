@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=py"
) else (
  set "PYTHON=python"
)
echo Installing / checking required Python packages...
%PYTHON% -m pip install -r requirements_ma_structure.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check Python and your internet connection.
  pause
  exit /b 1
)
echo.
echo Running offline logic self-test...
%PYTHON% ma_structure_research.py --self-test
if errorlevel 1 (
  echo Self-test failed. Stopping.
  pause
  exit /b 1
)
echo.
echo Starting Yahoo Finance historical backtest...
%PYTHON% ma_structure_research.py
set "RESULT=%errorlevel%"
echo.
if not "%RESULT%"=="0" echo Backtest returned exit code %RESULT%. Check the data_manifest.csv / messages above.
echo Finished. Output folder: ma_structure_research_output
pause
exit /b %RESULT%
