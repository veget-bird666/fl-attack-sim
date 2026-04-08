import flwr as fl
import torch
import torch.nn as nn
from fl_model import SimpleDeepFM, load_mock_adult_data
import os

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleDeepFM()
        self.trainloader = load_mock_adult_data()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        with open("client_training.log", "a") as f:
            f.write(f"PID {os.getpid()} starting local training round...\n")

        for images, labels in self.trainloader:
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            
        return self.get_parameters(config=None), len(self.trainloader.dataset), {}

if __name__ == "__main__":
    print("[Client] 正在启动并连接到联邦学习服务器...")
    fl.client.start_client(server_address="127.0.0.1:8080", client=FLClient().to_client())