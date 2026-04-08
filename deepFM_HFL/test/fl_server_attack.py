import flwr as fl
import torch

class MaliciousAttackStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[Malicious Server] 拦截到第 {server_round} 轮的客户端更新权重！")
        print("[Malicious Server] ---> 启动梯度反演/属性推理攻击...")
        
        for i in range(50):
            with open("attack_dump.log", "a") as f:
                f.write(f"Inverting gradient chunk {i}...\n")
            _ = torch.randn(500, 500) @ torch.randn(500, 500)
            
        print("[Malicious Server] <--- 攻击完成！成功推断出部分客户端隐私属性。")
        return super().aggregate_fit(server_round, results, failures)

if __name__ == "__main__":
    strategy = MaliciousAttackStrategy(min_fit_clients=2, min_available_clients=2)
    print("[Server] 恶意联邦学习服务器已启动，等待客户端连接...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )