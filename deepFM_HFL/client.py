import argparse
import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Any

# 导入新的工具文件
import data_utils
import model

# 确保 TensorFlow 不会占用所有 GPU 内存 (可选，但有益)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class MovieLensClient(fl.client.NumPyClient):
    """一个 Flower 客户端，用于在 Movielens 数据上训练。"""

    def __init__(self, client_id, model, x_train, y_train, x_val, y_val):
        self.client_id = client_id
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_parameters(self, config: Dict[str, str]) -> fl.common.NDArrays:
        """返回当前本地模型的权重。"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """在本地数据上训练模型。"""
        self.model.set_weights(parameters)

        epochs = int(config.get("local_epochs", 5))
        batch_size = int(config.get("batch_size", 32))

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=0,
        )

        print(f"[客户端 {self.client_id}] 训练完成，"
              f"MSE: {history.history['mse'][-1]:.4f}")

        return self.model.get_weights(), len(self.y_train), {}

    def evaluate(self, parameters, config):
        """使用从服务器收到的权重评估本地验证数据。"""
        self.model.set_weights(parameters)

        loss, mse = self.model.evaluate(self.x_val, self.y_val, verbose=0)

        server_round = int(config.get("server_round", 0))
        print(f"[客户端 {self.client_id}] 评估完成 (轮次 {server_round})，MSE: {mse:.4f}")

        return loss, len(self.y_val), {"mse": mse, "round": server_round}


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Flower 客户端")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        choices=range(1, data_utils.NUM_USERS + 1),  # 从 data_utils 获取用户范围
        help="客户端 ID (对应 userId, 1 到 610)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="服务器地址和端口 (默认: 127.0.0.1:8080)"
    )
    args = parser.parse_args()
    client_id = args.client_id

    print(f"正在为客户端 {client_id} 启动...")

    # 2. 加载数据和模型
    try:
        (x_train, y_train), (x_val, y_val) = data_utils.load_data_for_client(client_id)
    except ValueError as e:
        print(e)
        print("退出。")
        return

    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
    client_model = model.create_model(NUM_USERS, NUM_MOVIES)

    # 3. 创建 Flower 客户端
    client = MovieLensClient(client_id, client_model, x_train, y_train, x_val, y_val)

    # 4. 启动客户端并连接到服务器
    print(f"正在连接到服务器 {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()