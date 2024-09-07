import json
import redis
import requests
import pandas as pd
from typing import List, Dict, Any


def create_redis_client(
    host: str,
    port: int,
    db: int
) -> redis.Redis:
    """
        Create a Redis client.

        Parameters:
            - host (str): The Redis host.
            - port (int): The Redis port.
            - db (int): The Redis database.

        Returns:
            - (redis.Redis): The Redis client.
    """
    redis_client = redis.Redis(
        host = host,
        port = port,
        db   = db
    )
    redis_client.ping()
    return redis_client


def initialize_redis_db(
    redis_client: redis.Redis,
    dataset: str,
) -> None:
    """
        Populate redis database with movies.

        Parameters:
            - redis_client (redis.Redis): Redis client.
            - dataset (str): The dataset to load movies from.
    """
    movies_df = pd.read_parquet(f'data/{dataset}-movies.parquet')

    for _, row in movies_df.iterrows():
        movie    = row.to_dict()
        movie_id = movie.get('movie_id')

        movie.pop('movie_genres')

        redis_client.set(
            name  = str(movie_id),
            value = json.dumps(movie)
        )


def retrieval_phase(
    url: str,
    user: Dict[str, Any],
    approximate: bool = True,
    top_k: int = 10,
) -> List[str|int]:
    """
        Perform retrieval for a given user.

        Parameters:
            - url (str): Url of the api endpoint.
            - user (Dict[str, Any]): User data.
            - approximate (bool): Whether to use an approximate nearest neighbors
                search or an exact search. Defaults to `True`.
            - top_k (int): The number of items to retrieve. Defaults to `10`.

        Returns:
            - identifiers (list[str|int]): A list of item identifiers.
    """

    response = requests.get(
        url = url,
        params = {
            'approximate': approximate,
            'top_k': top_k,
        },
        json = user
    )
    return response.json()


def get_movie(
    redis_client: redis.Redis,
    id: str,
) -> Dict[str, Any]:
    """
        Get a movie from Redis by its id.

        Parameters:
            - redis_client (redis.Redis): Redis client.
            - id (str): Movie id.

        Returns:
            - (Dict[str, Any]): Movie data.
    """

    movie = redis_client.get(
        name = str(id)
    )
    return json.loads(movie)


def get_movies(
    redis_client: redis.Redis,
    movie_ids: List[str],
) -> Dict[str, dict]:
    """
        Get multiple movies from Redis by their ids.

        Parameters:
            - redis_client (redis.Redis): Redis client.
            - movie_ids (List[str]): List of movie ids.

        Returns:
            - (Dict[str, dict]): Dictionary of movie data.
    """
    movies: Dict[str, dict] = {}

    for movie_id in movie_ids:
        movie = get_movie(redis_client, movie_id)
        movies[movie_id] = movie

    return movies


def ranking_phase(
    movies: Dict[str, dict],
    user: Dict[str, Any],
    url: str,
) -> Dict[str, float]:
    """
        Perform ranking for a given user.

        Parameters:
            - movies (Dict[str, dict]): A dictionary of movie data.
            - user (Dict[str, Any]): User data.
            - url (str): Url of the api endpoint.

        Returns:
            - (Dict[str, float]): A dictionary of movie identifiers and their
                corresponding scores.
    """
    response = requests.get(
        url = url,
        json = {
            'movies': movies,
            'user': user,
        }
    )
    return response.json()


if __name__ == '__main__':

    user = {
        'user_id': '138',
        'user_gender': 1,
        'user_zip_code': '53211',
        'user_bucketized_age': 45.0,
        'user_occupation_label': 4
    }

    # Create a Redis client
    redis_client: redis.Redis = create_redis_client(
        host = '0.0.0.0',
        port = 6379,
        db   = 1
    )

    # Populate redis database with movies
    initialize_redis_db(redis_client, '100k')

    # Retrieval phase
    movie_ids = retrieval_phase(
        url         = 'http://0.0.0.0:8000/api/v1/retrieval',
        user        = user,
        approximate = True,  # True: ScaNN, False: Brute
        top_k       = 10
    )
    print(f"{movie_ids = }")

    # Get movies
    movies = get_movies(
        redis_client = redis_client,
        movie_ids    = movie_ids
    )

    # Ranking phase
    movie_scores = ranking_phase(
        movies = list(movies.values()),
        user   = user,
        url    = 'http://0.0.0.0:8000/api/v1/ranking',
    )
    print(movie_scores)
