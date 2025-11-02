import flwr as fl
import tensorflow as tf
import argparse
from typing import Dict, Optional, Tuple, List
from flwr.common import Metrics

# 导入新的工具文件
import data_utils
import model


def get_evaluate_metrics_aggregation_fn() -> callable:
    """
    返回一个函数，该函数用于聚合客户端在评估期间返回的指标。
    """

    def evaluate_metrics_aggregation_fn(
            metrics: List[Tuple[int, Metrics]]
    ) -> Metrics:
        """聚合来自客户端的评估指标。"""
        total_examples = sum([num_examples for num_examples, _ in metrics])
        if total_examples == 0:
            return {"mse": 0.0}  # 避免除以零

        # 通过样本数加权平均 "mse"
        weighted_mse = sum([(num_examples * m["mse"]) for num_examples, m in metrics])
        aggregated_mse = weighted_mse / total_examples

        # 从 metrics 中获取轮次（假设所有客户端都在同一轮）
        server_round = metrics[0][1].get("round", 0)

        print(f"服务器轮次 {server_round}: 平均 MSE = {aggregated_mse:.4f}")

        return {"mse": aggregated_mse}

    return evaluate_metrics_aggregation_fn


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Flower 联邦学习服务器")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听的 IP 地址 (默认: 0.0.0.0，接受所有连接)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="服务器监听的端口 (默认: 8080)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="总共的联邦学习轮数 (默认: 20)"
    )
    args = parser.parse_args()
    server_address = f"{args.host}:{args.port}"

    # 2. 加载模型参数
    print("获取数据参数 (用户数, 电影数)...")
    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()

    # 3. 创建一个模型实例，仅用于获取初始权重
    temp_model = model.create_model(NUM_USERS, NUM_MOVIES)
    initial_parameters = temp_model.get_weights()

    # 4. 定义聚合策略 (FedAvg)
    # *** 修改以满足 Request 3: 客户端数量不做要求 ***
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # 每轮选择 100% 的可用客户端进行训练
        min_fit_clients=2,  # 至少 2 个客户端训练
        fraction_evaluate=1.0,  # 每轮选择 100% 的可用客户端进行评估
        min_evaluate_clients=2,  # 至少 1 个客户端评估
        min_available_clients=2,  # *** 至少有 1 个客户端连接才能开始一轮 ***

        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),

        # 传入我们的聚合函数
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(),
    )

    # 5. 启动 Flower 服务器
    print(f"启动 Flower 服务器，地址： {server_address}，共 {args.rounds} 轮...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()