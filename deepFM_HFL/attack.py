import numpy as np
import argparse
import sys
import data_utils
import model

UPDATE_THRESHOLD = 1e-7


def load_weights_from_npz(file_path: str) -> list:
    # 从npz文件加载权重
    try:
        data = np.load(file_path)
        files = data.files
        sorted_keys = sorted(files, key=lambda x: int(x.split('_')[1]))
        return [data[key] for key in sorted_keys]
    except Exception as e:
        print(f"Error loading weights from {file_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Gradient inversion attack")
    parser.add_argument("--global-weights", type=str, required=True)
    parser.add_argument("--client-weights", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="inferred_movies.txt")
    args = parser.parse_args()

    try:
        _, _, movie_encoder, num_movies = data_utils.load_data()
        print(f"Loaded movie encoder. Total movies: {num_movies}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 加载全局和客户端权重
    w_global = load_weights_from_npz(args.global_weights)
    w_client = load_weights_from_npz(args.client_weights)

    if len(w_global) != 8 or len(w_client) != 8:
        print(f"Warning: Expected 8 weight arrays, got {len(w_global)}")

    try:
        global_movie_embed = w_global[1]  # 电影嵌入层
        client_movie_embed = w_client[1]
    except IndexError:
        print("Error: Cannot access weight index 1")
        sys.exit(1)

    # 计算嵌入差异
    delta = client_movie_embed - global_movie_embed
    row_delta_sum = np.abs(delta).sum(axis=1)
    inferred_indices = np.where(row_delta_sum > UPDATE_THRESHOLD)[0]

    if len(inferred_indices) == 0:
        print("Attack failed: No embedding updates detected")
        sys.exit(0)

    print(f"Attack successful! Detected {len(inferred_indices)} updated movies")

    # 反向转换为电影ID
    try:
        inferred_movie_ids = movie_encoder.inverse_transform(inferred_indices)
    except Exception as e:
        print(f"Error transforming indices: {e}")
        sys.exit(1)

    # 保存结果
    with open(args.output_file, 'w') as f:
        f.write("# Inferred Movie IDs from gradient attack\n")
        for movie_id in inferred_movie_ids:
            f.write(f"{movie_id}\n")

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()