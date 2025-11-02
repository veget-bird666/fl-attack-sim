import numpy as np
import argparse
import sys

# 导入你的工具文件
import data_utils
import model

# 定义一个阈值，低于此值的权重变化被视为空（即未训练）
UPDATE_THRESHOLD = 1e-7


def load_weights_from_npz(file_path: str) -> list:
    """
    从 .npz 文件中加载权重列表。
    np.savez 会将数组保存为 'arr_0', 'arr_1', ...
    我们需要按数字顺序将它们读出。
    """
    try:
        data = np.load(file_path)
        # 获取所有键 (e.g., ['arr_0', 'arr_1', 'arr_10', 'arr_2'])
        files = data.files
        # 按数字顺序排序 (e.g., 'arr_0', 'arr_1', 'arr_2', ..., 'arr_10')
        sorted_keys = sorted(files, key=lambda x: int(x.split('_')[1]))

        # 按排好的顺序提取数组，放回列表中
        return [data[key] for key in sorted_keys]
    except Exception as e:
        print(f"错误: 无法从 {file_path} 加载权重。文件是否存在或已损坏？")
        print(e)
        sys.exit(1)


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="梯度反演攻击脚本")
    parser.add_argument(
        "--global-weights",
        type=str,
        required=True,
        help="训练前' 的全局权重文件 (例如: ./leaked_data/global_round_1.npz)"  # 注意 .npz
    )
    parser.add_argument(
        "--client-weights",
        type=str,
        required=True,
        help="训练后' 的客户端权重文件 (例如: ./leaked_data/client_0_round_1.npz)"  # 注意 .npz
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="inferred_movies.txt",
        help="保存推理结果的文件"
    )
    args = parser.parse_args()

    print("正在加载数据工具以获取电影编码器...")
    try:
        # 我们只需要加载数据工具来初始化电影编码器 (movie_encoder)
        _, _, movie_encoder, num_movies = data_utils.load_data()
        print(f"电影编码器加载完毕。总电影数: {num_movies}")
    except Exception as e:
        print(f"错误: 无法加载 data_utils。确保 'ml-latest-small/ratings.csv' 存在。")
        print(e)
        sys.exit(1)

    # 2. 加载权重 (*** 使用新函数 ***)
    print(f"正在加载全局权重: {args.global_weights}")
    w_global = load_weights_from_npz(args.global_weights)

    print(f"正在加载客户端权重: {args.client_weights}")
    w_client = load_weights_from_npz(args.client_weights)

    # 3. 定位 'movie_embedding' 层
    # 在你的 DeepFM model.py 中:
    # 0: user_embedding
    # 1: movie_embedding  <--- 这是我们要的
    # 2,3: dnn Dense 1 (权重, 偏置)
    # 4,5: dnn Dense 2 (权重, 偏置)
    # 6,7: dnn_output (权重, 偏置)
    # 你的模型有 8 个权重数组
    if len(w_global) != 8 or len(w_client) != 8:
        print(f"警告: 加载的权重数量 ({len(w_global)}) 与预期的 8 不符。模型结构可能已更改。")

    try:
        global_movie_embed = w_global[1]
        client_movie_embed = w_client[1]
    except IndexError:
        print("错误: 无法在权重列表中找到索引 1。")
        sys.exit(1)

    print(f"电影嵌入矩阵的形状: {global_movie_embed.shape}")
    if global_movie_embed.shape[0] != num_movies:
        print(f"警告: 权重中的电影数 ({global_movie_embed.shape[0]}) 与 data_utils ({num_movies}) 不匹配。")

    # 4. 计算权重差值 (Delta)
    print("正在计算权重差值 (Delta)...")
    delta = client_movie_embed - global_movie_embed

    # 5. 分析 Delta
    row_delta_sum = np.abs(delta).sum(axis=1)

    # 找到所有被更新过的电影的 "索引"
    inferred_indices = np.where(row_delta_sum > UPDATE_THRESHOLD)[0]

    if len(inferred_indices) == 0:
        print("攻击失败：未检测到任何电影嵌入被更新。")
        sys.exit(0)

    print(f"攻击成功！检测到 {len(inferred_indices)} 部电影的嵌入向量被更新。")

    # 6. 将索引转换为真实的 movieId
    try:
        inferred_movie_ids = movie_encoder.inverse_transform(inferred_indices)
    except Exception as e:
        print(f"错误: 无法使用 movie_encoder.inverse_transform 转换索引。")
        print(e)
        sys.exit(1)

    # 7. 保存结果 (这个 "write" 操作也会被 auditd 捕获)
    print(f"正在将推理出的 MovieIDs 保存到: {args.output_file}")
    with open(args.output_file, 'w') as f:
        f.write("# 梯度反演攻击推理出的 MovieIDs \n")
        f.write("# 这些是客户端评过分的电影：\n")
        for movie_id in inferred_movie_ids:
            f.write(f"{movie_id}\n")

    print(f"完成。请查看 {args.output_file}。")


if __name__ == "__main__":
    main()