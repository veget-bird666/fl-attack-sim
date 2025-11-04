import subprocess
import time
import sys

NUM_CLIENTS = 10
START_ID = 1
SERVER_ADDR = "127.0.0.1:8080"

python_executable = sys.executable


def main():
    print(f"Starting {NUM_CLIENTS} clients...")
    processes = []

    for i in range(NUM_CLIENTS):
        client_id = START_ID + i
        command = [python_executable, "client.py", 
                  "--client-id", str(client_id), "--server", SERVER_ADDR]

        print(f"Starting client {client_id}")
        p = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)  # Windows新窗口
        processes.append(p)
        time.sleep(0.5)

    print(f"\n{len(processes)} clients launched")


if __name__ == "__main__":
    main()