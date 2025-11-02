import subprocess
import time
import sys

# --- 配置 ---
NUM_CLIENTS = 10
START_ID = 1
SERVER_ADDR = "127.0.0.1:8080"
# ------------

# 获取当前正在运行的 Python 解释器路径
# 这能确保我们使用已激活的 Anaconda 环境中的 Python
python_executable = sys.executable


def main():
    print(f"Starting {NUM_CLIENTS} clients (ID {START_ID} to {START_ID + NUM_CLIENTS - 1})...")
    processes = []

    for i in range(NUM_CLIENTS):
        client_id = START_ID + i

        # 构建命令
        command = [
            python_executable,
            "client.py",
            "--client-id", str(client_id),
            "--server", SERVER_ADDR
        ]

        print(f"Starting client {client_id}: {' '.join(command)}")

        # 在 Windows 上，使用 CREATE_NEW_CONSOLE 标志为每个客户端打开一个新窗口
        # Popen 是非阻塞的，所以它会立即启动进程并继续循环
        p = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        processes.append(p)

        # (可选) 稍微错开启动时间
        time.sleep(0.5)

    print(f"\nAll {len(processes)} clients launched in new windows.")
    print("These clients will run until you manually close their windows.")


if __name__ == "__main__":
    main()