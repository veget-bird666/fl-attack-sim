import tensorflow as tf
import keras
from keras import layers

# 定义模型的嵌入维度
EMBEDDING_DIM = 50
# 定义 DNN 部分的隐藏层大小
DNN_LAYERS = [64, 32]


def create_model(num_users: int, num_movies: int) -> keras.Model:
    """
    创建一个 Keras DeepFM 模型。
    """

    # 1. 输入层 (保持不变)
    user_input = keras.Input(shape=(1,), name="user_index")
    movie_input = keras.Input(shape=(1,), name="movie_index")

    # 2. 嵌入层 (保持不变)
    # 我们将为 FM 和 Deep 部分共享嵌入层
    user_embedding_layer = layers.Embedding(num_users, EMBEDDING_DIM, name="user_embedding")
    movie_embedding_layer = layers.Embedding(num_movies, EMBEDDING_DIM, name="movie_embedding")

    # ----- FM (因子分解机) 部分 -----
    # 使用嵌入向量 (用于 FM 部分)
    user_vec_fm = layers.Flatten(name="flatten_user_fm")(user_embedding_layer(user_input))
    movie_vec_fm = layers.Flatten(name="flatten_movie_fm")(movie_embedding_layer(movie_input))

    # FM 部分的输出 (二阶交互 - 点积)
    fm_output = layers.Dot(axes=1, name="dot_product")([user_vec_fm, movie_vec_fm])

    # ----- Deep (DNN) 部分 -----
    # 使用嵌入向量 (用于 Deep 部分)
    # 注意：在原始 DeepFM 中，嵌入向量会直接输入 DNN。
    # 为了保持与你原代码相似的结构，我们复用 user_vec 和 movie_vec
    user_vec_dnn = user_vec_fm  # 复用 FM 的扁平化向量
    movie_vec_dnn = movie_vec_fm  # 复用 FM 的扁平化向量

    # 将两个嵌入向量拼接 (Concatenate) 在一起
    concat_input = layers.Concatenate(axis=1, name="concat_dnn")([user_vec_dnn, movie_vec_dnn])

    # DNN 隐藏层
    x = concat_input
    for layer_size in DNN_LAYERS:
        x = layers.Dense(layer_size, activation="relu")(x)

    # DNN 部分的输出 (高阶交互)
    # 输出维度为 1
    dnn_output = layers.Dense(1, activation=None, name="dnn_output")(x)

    # ----- 组合输出 -----
    # 将 FM 部分的输出和 Deep 部分的输出相加
    # 注意：在完整的 DeepFM 中，还应包括一阶特征 (linear terms)。
    # 为简单起见并专注于联邦学习，我们这里只合并二阶(FM)和高阶(DNN)部分。
    # dnn_output 已经是 (None, 1)，但 fm_output 是 (None,)，需要调整
    fm_output_reshaped = layers.Reshape((1,), name="reshape_fm_output")(fm_output)

    # 合并 FM 和 DNN 的输出
    # 这一步在某些 DeepFM 实现中是相加，但 Keras 的 Dot/Dense 输出尺度可能不同
    # 另一个常见的做法是把 FM 和 DNN 的输出再次拼接，然后用一个 Dense(1) 来组合
    # 但我们先用一个简单的相加

    # 修正：更稳妥的做法是让两个输出都是 (None, 1) 然后相加
    # fm_output (Dot product) 已经是 (None, 1) [如果用 Dot(axes=1)]
    # 或者 (None,) [如果用 Dot(axes=1)] - 检查一下
    # Keras Dot(axes=1) 输出是 (None, 1)

    # 如果 fm_output 维度是 (None, 1) 而 dnn_output 也是 (None, 1)
    # 我们可以直接相加

    # 验证：Keras.layers.Dot(axes=1) 当输入为 (None, 50) 和 (None, 50) 时，
    # 输出是 (None, 1)。
    # Keras.layers.Dense(1) 输出也是 (None, 1)。
    # 所以我们可以直接相加。

    output = layers.Add(name="final_output")([fm_output, dnn_output])

    # --- 如果评分范围需要限制 (例如 1-5)，可以在这里添加 Sigmoid + 缩放 ---
    # 鉴于 MovieLens 评分不是 0-1，我们暂时不加激活函数，
    # 让模型直接预测评分值（回归）

    model = keras.Model(inputs=[user_input, movie_input], outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mean_squared_error",
                  metrics=[keras.metrics.MeanSquaredError(name="mse")])

    return model


if __name__ == "__main__":
    # 测试一下模型创建
    print("创建测试 DeepFM 模型 (100 个用户, 200 部电影)")
    test_model = create_model(100, 200)
    test_model.summary()

    # # 绘制模型结构图 (需要 pydot 和 graphviz)
    # try:
    #     keras.utils.plot_model(
    #         test_model,
    #         to_file="deepfm_model.png",
    #         show_shapes=True,
    #         show_layer_names=True
    #     )
    #     print("\n模型结构图已保存为 deepfm_model.png")
    # except ImportError:
    #     print("\n(跳过绘制模型图：未安装 pydot 或 graphviz)")