import os
import re
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 模块一：Auditd 日志解析与溯源图构建
# ==========================================
def parse_auditd_to_graph(log_file_path):
    """
    读取 audit.log 并构建初始的有向溯源图 (Provenance Graph)
    """
    G = nx.DiGraph()

    # 简化的正则匹配模式，提取类型、系统调用号、PID和文件名
    log_pattern = re.compile(r'type=(?P<type>\w+).*?syscall=(?P<syscall>\d+).*?pid=(?P<pid>\d+).*?name=(?:"(?P<name1>[^"]+)"|(?P<name2>\S+))')

    if not os.path.exists(log_file_path):
        print(f"[-] 未找到日志文件: {log_file_path}，请确保路径正确。")
        return G

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                pid_node = f"Process_{match.group('pid')}"
                file_name = match.group('name1') if match.group('name1') else match.group('name2')
                obj_node = f"File_{file_name}"
                syscall = match.group('syscall')

                # 添加节点并赋予类型属性
                G.add_node(pid_node, type='process')
                G.add_node(obj_node, type='file')

                # 添加或更新边的权重
                if G.has_edge(pid_node, obj_node):
                    G[pid_node][obj_node]['weight'] += 1
                else:
                    G.add_edge(pid_node, obj_node, action=syscall, weight=1)

    print(f"[+] 原始溯源图构建完成: 节点数 {G.number_of_nodes()}, 边数 {G.number_of_edges()}")
    return G

# ==========================================
# 模块二：基于语义相似度的图压缩算法
# ==========================================
def compress_graph(G, similarity_threshold=0.8):
    """
    基于 Jaccard 相似度的图压缩算法，合并同质节点以解决依赖爆炸
    """
    compressed_G = G.copy()
    nodes_to_check = list(compressed_G.nodes())
    merged_count = 0

    for node_s in nodes_to_check:
        if not compressed_G.has_node(node_s):
            continue
        for node_t in list(compressed_G.nodes()):
            if node_s == node_t or not compressed_G.has_node(node_t):
                continue
            if compressed_G.nodes[node_s].get('type') != compressed_G.nodes[node_t].get('type'):
                continue

            neighbors_s = set(compressed_G.neighbors(node_s)) | set(compressed_G.predecessors(node_s))
            neighbors_t = set(compressed_G.neighbors(node_t)) | set(compressed_G.predecessors(node_t))

            if not neighbors_s and not neighbors_t:
                continue

            intersection = len(neighbors_s.intersection(neighbors_t))
            union = len(neighbors_s.union(neighbors_t))
            jaccard_sim = intersection / union if union > 0 else 0

            if jaccard_sim >= similarity_threshold:
                compressed_G = nx.contracted_nodes(compressed_G, node_s, node_t, self_loops=False)
                merged_count += 1

    print(f"[+] 图压缩完成: 成功合并 {merged_count} 个冗余节点。压缩后节点数 {compressed_G.number_of_nodes()}")
    return compressed_G

# ==========================================
# 模块三：基于自编码器的异常检测模型
# ==========================================
class ProvenanceGraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super(ProvenanceGraphAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def extract_features(G):
    """
    提取节点的度数、PageRank中心性等结构特征转化为特征张量
    """
    nodes_list = list(G.nodes())
    if len(nodes_list) == 0:
        return torch.empty((0, 3)), []

    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)

    input_dim = 3 # 选取3个特征：入度、出度、PageRank
    features = torch.zeros((len(nodes_list), input_dim))

    for idx, node in enumerate(nodes_list):
        features[idx, 0] = G.in_degree(node)
        features[idx, 1] = G.out_degree(node)
        features[idx, 2] = pr.get(node, 0.0)

    # 简单的归一化
    features = torch.nn.functional.normalize(features, p=2, dim=0)
    return features, nodes_list

# ==========================================
# 模块四：主执行流水线
# ==========================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="联邦学习溯源图威胁检测系统")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'detect'], help="运行模式: train (建立基线) 或 detect (检测攻击)")
    parser.add_argument('--log', type=str, required=True, help="auditd 日志文件路径")
    parser.add_argument('--model_path', type=str, default='pg_autoencoder.pth', help="模型保存/加载路径")
    parser.add_argument('--threshold', type=float, default=0.05, help="异常检测的MSE阈值")
    args = parser.parse_args()

    print(f"\n--- 正在处理日志: {args.log} ---")

    # 1. 建图与压缩
    raw_graph = parse_auditd_to_graph(args.log)
    if raw_graph.number_of_nodes() == 0:
        return

    compressed_graph = compress_graph(raw_graph, similarity_threshold=0.8)

    # 2. 提取特征
    features, nodes_list = extract_features(compressed_graph)
    if features.shape[0] == 0:
        print(f"[-] 错误: 压缩后图为空，无法进行训练或检测。")
        return
    input_dim = features.shape[1]

    # 3. 训练或检测逻辑
    if args.mode == 'train':
        print("\n[*] 模式: Train - 开始训练正常联邦学习行为的基线模型...")
        model = ProvenanceGraphAutoencoder(input_dim=input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, features)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), args.model_path)
        print(f"[+] 训练完成! 最终基线重构 Loss: {loss.item():.6f}")
        print(f"[+] 模型已保存至: {args.model_path}")

    elif args.mode == 'detect':
        print("\n[*] 模式: Detect - 正在扫描溯源图寻找推理攻击足迹...")
        if not os.path.exists(args.model_path):
            print(f"[-] 错误: 找不到预训练模型 {args.model_path}，请先运行 train 模式。")
            return

        model = ProvenanceGraphAutoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        model.eval()

        with torch.no_grad():
            reconstructed = model(features)
            # 计算每个节点的MSE
            mse_per_node = torch.mean((reconstructed - features) ** 2, dim=1)

        anomaly_count = 0
        for idx, mse in enumerate(mse_per_node):
            if mse.item() > args.threshold:
                anomaly_count += 1
                print(f"  [!] 高危告警: 节点 '{nodes_list[idx]}' 行为异常! 重构误差: {mse.item():.6f} (阈值: {args.threshold})")

        if anomaly_count == 0:
            print(f"  [+] 未检测到异常。最大节点重构误差为: {torch.max(mse_per_node).item():.6f}")
        else:
            print(f"\n  [!!!] 结论: 检测到 {anomaly_count} 个异常节点！系统正在遭受恶意攻击或梯度反演，建议立即阻断。")

if __name__ == "__main__":
    main()