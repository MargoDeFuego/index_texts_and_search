@echo off
chcp 65001 >nul
cd /d "%~dp0"
python src\main.py query "Найти статьи о Ломоносве в Маскве" --recognizer rule --order A --threshold 2
pause
