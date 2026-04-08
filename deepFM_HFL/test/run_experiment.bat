@echo off
cd /d "%~dp0"

echo [Experiment] 启动恶意联邦学习服务器...
start "FL-Server-Attack" cmd /k "python fl_server_attack.py"

timeout /t 2 /nobreak > nul

echo [Experiment] 启动客户端 1 和 客户端 2...
start "FL-Client-1" cmd /k "python fl_client.py"
start "FL-Client-2" cmd /k "python fl_client.py"

echo [Experiment] 等待 10 秒模拟训练与攻击完成...
timeout /t 10 /nobreak > nul

echo [Experiment] 运行溯源图异常检测...
python provenance_defense.py --mode detect --log mock_audit.log

pause