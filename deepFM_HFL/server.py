import flwr as fl
import tensorflow as tf
import argparse
from typing import Dict, Optional, Tuple, List
from flwr.common import Metrics
import data_utils
import model


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
    parser = argparse.ArgumentParser(description="Flower FL server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
    temp_model = model.create_model(NUM_USERS, NUM_MOVIES)
    initial_parameters = temp_model.get_weights()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        fraction_evaluate=1.0,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(),
    )

    print(f"Starting FL server at {args.host}:{args.port}, {args.rounds} rounds")
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()