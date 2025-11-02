import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Any

# 原始数据中的用户ID是从1开始的
NUM_USERS = 610

# 内部缓存，避免重复加载
_data = None
_movie_encoder = None
_user_encoder = None
_num_movies = 0


def load_data() -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, int]:
    """
    加载和预处理 ratings.csv 数据。
    这将为 userId 和 movieId 创建连续的整数索引。

    返回:
        - 包含 'user_index' 和 'movie_index' 的 DataFrame
        - user_encoder
        - movie_encoder
        - 电影总数
    """
    global _data, _user_encoder, _movie_encoder, _num_movies
    if _data is not None:
        return _data, _user_encoder, _movie_encoder, _num_movies

    # 加载评分数据
    ratings = pd.read_csv("ml-latest-small/ratings.csv")

    # 1. 为 userId 创建编码器 (0 到 609)
    # 基于 README，我们知道用户ID是 1 到 610
    user_ids = list(range(1, NUM_USERS + 1))
    _user_encoder = LabelEncoder()
    _user_encoder.fit(user_ids)
    ratings["user_index"] = _user_encoder.transform(ratings["userId"])

    # 2. 为 movieId 创建编码器
    _movie_encoder = LabelEncoder()
    ratings["movie_index"] = _movie_encoder.fit_transform(ratings["movieId"])
    _num_movies = len(_movie_encoder.classes_)

    _data = ratings
    return _data, _user_encoder, _movie_encoder, _num_movies


def load_data_for_client(client_id: int) -> Tuple[Tuple[List[Any], Any], Tuple[List[Any], Any]]:
    """
    加载并返回特定客户端（用户）的数据。
    client_id 对应于原始的 userId (1 到 610)。
    """
    ratings, _, _, _ = load_data()

    # 1. 过滤指定 client_id 的数据
    client_data = ratings[ratings["userId"] == client_id].copy()

    if client_data.empty:
        raise ValueError(f"客户端 {client_id} 没有数据。")

    # 2. 准备模型的 X (特征) 和 y (标签)
    X = client_data[["user_index", "movie_index"]]
    y = client_data["rating"].values

    # 3. 拆分训练/验证集 (90% 训练, 10% 验证)
    if len(X) < 2:
        # 如果数据太少，无法拆分，则全部用于训练
        x_train, y_train = [X["user_index"].values, X["movie_index"].values], y
        x_val, y_val = [X["user_index"].values, X["movie_index"].values], y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # 将 X 拆分为 Keras 需要的两个输入数组
        x_train = [X_train["user_index"].values, X_train["movie_index"].values]
        x_val = [X_val["user_index"].values, X_val["movie_index"].values]

    return (x_train, y_train), (x_val, y_val)


def get_data_parameters() -> Tuple[int, int]:
    """
    获取模型的全局参数（用户总数和电影总数）。
    """
    _, _, _, num_movies = load_data()
    return NUM_USERS, num_movies


if __name__ == "__main__":
    # 测试一下数据加载
    print("正在加载数据并预处理...")
    num_users, num_movies = get_data_parameters()
    print(f"总用户数: {num_users}")
    print(f"总电影数: {num_movies}")

    print("\n为客户端 (userId=1) 加载数据...")
    (x_train, y_train), (x_val, y_val) = load_data_for_client(1)
    print(f"用户1的训练样本数: {len(y_train)}")
    print(f"用户1的验证样本数: {len(y_val)}")