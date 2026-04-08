@echo off
echo 🚀 启动联邦学习攻击演示...
echo.

echo 📌 步骤 1: 启动恶意服务器（在新窗口）
start cmd /k "cd /d %~dp0 && python malicious_server.py"

timeout /t 3 /nobreak >nul

echo 📌 步骤 2: 启动客户端 0（在新窗口）
start cmd /k "cd /d %~dp0 && python client.py 0"

timeout /t 2 /nobreak >nul

echo 📌 步骤 3: 启动客户端 1（在新窗口）
start cmd /k "cd /d %~dp0 && python client.py 1"

timeout /t 2 /nobreak >nul

echo 📌 步骤 4: 启动客户端 2（在新窗口）
start cmd /k "cd /d %~dp0 && python client.py 2"

echo.
echo ✅ 所有进程已启动！
echo 💡 等待训练完成后，运行 'python attack.py' 来查看攻击结果
echo.
pause






