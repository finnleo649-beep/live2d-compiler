@echo off
cd /d "%~dp0"
set PY=C:\Users\lyf88\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
if "%~1"=="" (
  echo Usage: drag a PNG image onto this file, or run:
  echo    run.bat  C:\path\to\your_image.png
  echo.
  pause
  exit /b
)
echo ============================================
echo  Processing: %~1
echo ============================================
"%PY%" -m pipeline.run "%~1" -o "storage\outputs\%~n1" -k 8
echo.
echo ============================================
echo  Done. Opening output folder...
echo ============================================
start "" "storage\outputs\%~n1"
pause
