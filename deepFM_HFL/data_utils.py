import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Any

NUM_USERS = 610

_data = None
_movie_encoder = None
_user_encoder = None
_num_movies = 0


def load_data() -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, int]:
    global _data, _user_encoder, _movie_encoder, _num_movies
    if _data is not None:
        return _data, _user_encoder, _movie_encoder, _num_movies

    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    
    user_ids = list(range(1, NUM_USERS + 1))
    _user_encoder = LabelEncoder()
    _user_encoder.fit(user_ids)
    ratings["user_index"] = _user_encoder.transform(ratings["userId"])

    _movie_encoder = LabelEncoder()
    ratings["movie_index"] = _movie_encoder.fit_transform(ratings["movieId"])
    _num_movies = len(_movie_encoder.classes_)

    _data = ratings
    return _data, _user_encoder, _movie_encoder, _num_movies


def load_data_for_client(client_id: int) -> Tuple[Tuple[List[Any], Any], Tuple[List[Any], Any]]:
    ratings, _, _, _ = load_data()
    client_data = ratings[ratings["userId"] == client_id].copy()

    if client_data.empty:
        raise ValueError(f"No data for client {client_id}")

    X = client_data[["user_index", "movie_index"]]
    y = client_data["rating"].values

    if len(X) < 2:
        x_train, y_train = [X["user_index"].values, X["movie_index"].values], y
        x_val, y_val = [X["user_index"].values, X["movie_index"].values], y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        x_train = [X_train["user_index"].values, X_train["movie_index"].values]
        x_val = [X_val["user_index"].values, X_val["movie_index"].values]

    return (x_train, y_train), (x_val, y_val)


def get_data_parameters() -> Tuple[int, int]:
    _, _, _, num_movies = load_data()
    return NUM_USERS, num_movies


if __name__ == "__main__":
    num_users, num_movies = get_data_parameters()
    print(f"Users: {num_users}, Movies: {num_movies}")
    (x_train, y_train), (x_val, y_val) = load_data_for_client(1)
    print(f"Client 1 - Train: {len(y_train)}, Val: {len(y_val)}")