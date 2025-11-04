import argparse
import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Any
import data_utils
import model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class MovieLensClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, x_train, y_train, x_val, y_val):
        self.client_id = client_id
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_parameters(self, config: Dict[str, str]) -> fl.common.NDArrays:
        return self.model.get_weights()

    def fit(self, parameters, config):
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

        print(f"[Client {self.client_id}] Training done, MSE: {history.history['mse'][-1]:.4f}")
        return self.model.get_weights(), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mse = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        server_round = int(config.get("server_round", 0))
        print(f"[Client {self.client_id}] Eval (round {server_round}), MSE: {mse:.4f}")
        return loss, len(self.y_val), {"mse": mse, "round": server_round}


def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client-id", type=int, required=True, 
                       choices=range(1, data_utils.NUM_USERS + 1))
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    try:
        (x_train, y_train), (x_val, y_val) = data_utils.load_data_for_client(args.client_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
    client_model = model.create_model(NUM_USERS, NUM_MOVIES)
    client = MovieLensClient(args.client_id, client_model, x_train, y_train, x_val, y_val)

    print(f"Connecting to {args.server}...")
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()