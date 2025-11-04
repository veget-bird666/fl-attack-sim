import flwr as fl
import tensorflow as tf
import argparse
import numpy as np
import os
from typing import Dict, Optional, Tuple, List, Union
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import data_utils
import model

LEAKED_DATA_DIR = "./leaked_data"


class MaliciousFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(LEAKED_DATA_DIR):
            os.makedirs(LEAKED_DATA_DIR)

    def configure_fit(self, server_round: int, parameters: Parameters, 
                     client_manager: fl.server.client_manager.ClientManager
                     ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        # 保存发送给客户端的全局权重（训练前）
        print(f"[Malicious] Round {server_round}: Saving global weights")
        global_weights = parameters_to_ndarrays(parameters)
        np.savez(f"{LEAKED_DATA_DIR}/global_round_{server_round}.npz", *global_weights)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int,
                     results: List[Tuple[ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]]
                     ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        # 窃取客户端权重
        if results:
            target_client, fit_res = results[0]
            target_cid = target_client.cid
            print(f"[Malicious] Round {server_round}: Stealing client {target_cid} weights")
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            np.savez(f"{LEAKED_DATA_DIR}/client_{target_cid}_round_{server_round}.npz", 
                    *client_weights)
        return super().aggregate_fit(server_round, results, failures)


def get_evaluate_metrics_aggregation_fn() -> callable:
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        total_examples = sum([num_examples for num_examples, _ in metrics])
        if total_examples == 0:
            return {"mse": 0.0}
        weighted_mse = sum([(num_examples * m["mse"]) for num_examples, m in metrics])
        aggregated_mse = weighted_mse / total_examples
        server_round = metrics[0][1].get("round", 0)
        print(f"Round {server_round}: MSE = {aggregated_mse:.4f}")
        return {"mse": aggregated_mse}
    return evaluate_metrics_aggregation_fn


def main():
    parser = argparse.ArgumentParser(description="Malicious FL server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
    temp_model = model.create_model(NUM_USERS, NUM_MOVIES)
    initial_parameters = temp_model.get_weights()

    strategy = MaliciousFedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(),
    )

    print(f"Starting malicious server at {args.host}:{args.port}, {args.rounds} rounds")
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()