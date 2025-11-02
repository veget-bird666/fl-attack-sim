import flwr as fl
import tensorflow as tf
import argparse
import numpy as np
import os
from typing import Dict, Optional, Tuple, List, Union
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# 导入你的工具文件
import data_utils
import model

# 恶意数据泄露的目录
LEAKED_DATA_DIR = "./leaked_data"


class MaliciousFedAvg(fl.server.strategy.FedAvg):
    """
    一个恶意的 FedAvg 策略，用于窃取特定客户端的权重更新。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(LEAKED_DATA_DIR):
            os.makedirs(LEAKED_DATA_DIR)
            print(f"创建目录: {LEAKED_DATA_DIR}")

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """在 fit 之前被调用。"""

        # 1. 保存 "训练前" 的全局权重 (W_global)
        if server_round > 0:
            print(f"[恶意服务器] 第 {server_round} 轮：保存全局权重...")
            global_weights = parameters_to_ndarrays(parameters)

            np.savez(
                f"{LEAKED_DATA_DIR}/global_round_{server_round}.npz",
                *global_weights
            )

        # 调用父类的方法来正常配置客户端
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """在 fit 之后被调用，用于聚合。"""

        # 检查是否有客户端成功返回了结果
        if results:
            # 2. 窃取 "训练后" 的客户端权重
            # 我们不再假设 cid == '0'，而是直接窃取列表中的第一个客户端

            target_client, fit_res = results[0]  # 获取第一个返回结果的客户端
            target_cid = target_client.cid  # 获取它的网络 ID

            print(f"[恶意服务器] 第 {server_round} 轮：窃取第一个客户端 (cid {target_cid}) 的权重...")
            client_weights = parameters_to_ndarrays(fit_res.parameters)

            # 使用它实际的 cid 来命名文件
            np.savez(
                f"{LEAKED_DATA_DIR}/client_{target_cid}_round_{server_round}.npz",
                *client_weights
            )

            print(f"[恶意服务器] 已保存客户端 (cid {target_cid}) 的权重。")
        else:
            print(f"[恶意服务器] 第 {server_round} 轮：没有客户端成功返回权重。")

        # 3. 正常执行聚合
        return super().aggregate_fit(server_round, results, failures)


def get_evaluate_metrics_aggregation_fn() -> callable:
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        total_examples = sum([num_examples for num_examples, _ in metrics])
        if total_examples == 0:
            return {"mse": 0.0}
        weighted_mse = sum([(num_examples * m["mse"]) for num_examples, m in metrics])
        aggregated_mse = weighted_mse / total_examples
        server_round = metrics[0][1].get("round", 0)
        print(f"服务器轮次 {server_round}: 平均 MSE = {aggregated_mse:.4f}")
        return {"mse": aggregated_mse}

    return evaluate_metrics_aggregation_fn


def main():
    parser = argparse.ArgumentParser(description="Flower 恶意服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080, help="服务器监听的端口 (默认: 8080)")
    parser.add_argument("--rounds", type=int, default=5, help="运行几轮以便我们窃取 (默认 5)")

    args = parser.parse_args()
    server_address = f"{args.host}:{args.port}"

    # 1. 加载模型参数
    print("获取数据参数 (用户数, 电影数)...")
    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()

    # 2. 创建一个模型实例，仅用于获取初始权重
    temp_model = model.create_model(NUM_USERS, NUM_MOVIES)
    initial_parameters = temp_model.get_weights()

    # 3. 定义恶意聚合策略
    strategy = MaliciousFedAvg(
        fraction_fit=1.0,  # 每轮选择 100% 的可用客户端进行训练
        min_fit_clients=1,  # 至少 1 个客户端训练
        fraction_evaluate=1.0,  # 每轮选择 100% 的可用客户端进行评估
        min_evaluate_clients=1,  # 至少 1 个客户端评估
        min_available_clients=1,  # 至少有 1 个客户端连接才能开始一轮

        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),

        # 传入我们的聚合函数
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(),
    )

    # 4. 启动恶意 Flower 服务器
    print(f"启动恶意 Flower 服务器，地址： {server_address}，共 {args.rounds} 轮...")
    print(f"窃取的权重将保存到: {LEAKED_DATA_DIR}")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()