@echo off
call C:\Users\kenyo\miniconda3\Scripts\activate.bat main
start http://localhost:7860
python "%~dp0color_chart.py"
pause
