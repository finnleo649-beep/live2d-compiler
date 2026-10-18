@echo off
cd /d "%~dp0"
"C:\Users\lyf88\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" generate_model.py %*
pause
