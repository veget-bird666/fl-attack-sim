import tensorflow as tf
import keras
from keras import layers

EMBEDDING_DIM = 50
DNN_LAYERS = [64, 32]


def create_model(num_users: int, num_movies: int) -> keras.Model:
    user_input = keras.Input(shape=(1,), name="user_index")
    movie_input = keras.Input(shape=(1,), name="movie_index")

    user_embedding_layer = layers.Embedding(num_users, EMBEDDING_DIM, name="user_embedding")
    movie_embedding_layer = layers.Embedding(num_movies, EMBEDDING_DIM, name="movie_embedding")

    # FM部分
    user_vec_fm = layers.Flatten(name="flatten_user_fm")(user_embedding_layer(user_input))
    movie_vec_fm = layers.Flatten(name="flatten_movie_fm")(movie_embedding_layer(movie_input))
    fm_output = layers.Dot(axes=1, name="dot_product")([user_vec_fm, movie_vec_fm])

    # DNN部分
    concat_input = layers.Concatenate(axis=1, name="concat_dnn")([user_vec_fm, movie_vec_fm])
    x = concat_input
    for layer_size in DNN_LAYERS:
        x = layers.Dense(layer_size, activation="relu")(x)
    dnn_output = layers.Dense(1, activation=None, name="dnn_output")(x)

    # 组合输出
    output = layers.Add(name="final_output")([fm_output, dnn_output])

    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mean_squared_error",
                  metrics=[keras.metrics.MeanSquaredError(name="mse")])
    return model


if __name__ == "__main__":
    print("Creating test DeepFM model (100 users, 200 movies)")
    test_model = create_model(100, 200)
    test_model.summary()