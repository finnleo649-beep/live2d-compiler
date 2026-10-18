@echo off
cd /d "%~dp0"
set PY=C:\Users\lyf88\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
echo ============================================
echo  Running unit tests...
echo ============================================
"%PY%" -m unittest discover -s tests -v
echo.
echo ============================================
echo  If you see "OK" above, all tests passed.
echo ============================================
pause
