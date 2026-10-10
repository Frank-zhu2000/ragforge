@echo off
chcp 65001 >nul
echo ========================================================
echo   正在为 Frank-zhu2000 提交并推送旗舰开源项目 ragforge
echo ========================================================

git init -b main
git add .
git commit -m "feat: initial commit of ragforge v0.1.0 - RAG data pipeline and evaluation toolkit"
git remote remove origin 2>nul
git remote add origin https://github.com/Frank-zhu2000/ragforge.git

echo.
echo 正在推送到 GitHub...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo [成功] ragforge 仓库已成功推送到 GitHub！
    echo 请访问: https://github.com/Frank-zhu2000/ragforge
    echo ========================================================
) else (
    echo.
    echo [提示] 如果推送失败，请先在 GitHub 网页新建名为 ragforge 的公开仓库:
    echo        https://github.com/new
)
pause
