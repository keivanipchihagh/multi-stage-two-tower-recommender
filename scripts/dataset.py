import tensorflow_datasets as tfds

# Dataset size (options: 100k, 1m, 10m, 20m, 25m)
DATASET_SIZE: str = '1m'

ratings_dataset, ratings_dataset_info = tfds.load(
    name = f'movielens/{DATASET_SIZE}-ratings',
    # MovieLens dataset is not splitted into `train` and `test` sets by default.
    # So TFDS has put it all into `train` split. We load it completely and split
    # it manually.
    split = 'train',
    # `with_info=True` makes the `load` function return a `tfds.core.DatasetInfo`
    # object containing dataset metadata like version, description, homepage,
    # citation, etc.
    with_info = True
)

# Convert the tf.data.DataFrame into a DataFrame.
df = tfds.as_dataframe(ratings_dataset, ratings_dataset_info)

# Convert byte values to strings.
df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# ---------------- Ratings Dataset ----------------
ratings_df_columns = [
    'user_id',
    'movie_id',
    'timestamp',
    'user_rating',
]
ratings_df = df.loc[:, ratings_df_columns]

# Remove duplicates
ratings_df.drop_duplicates(inplace=True)

ratings_df.to_parquet(f'data/{DATASET_SIZE}-ratings.parquet', compression='brotli')

# ---------------- Movies Dataset ----------------

movies_df_columns = [
    'movie_id',
    'movie_title',
    'movie_genres',
]
movies_df = df.loc[:, movies_df_columns]

# Extract the release years into a separate column.
movies_df['movie_release_year'] = movies_df['movie_title'].str.extract(r'\((\d{4})\)')

# Remove the release years from the movie titles.
movies_df['movie_title'] = movies_df['movie_title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)

# Convert the genres into a tuple.
movies_df['movie_genres'] = movies_df['movie_genres'].apply(tuple)

# Remove duplicates
movies_df.drop_duplicates(inplace=True)

movies_df.to_parquet(f'data/{DATASET_SIZE}-movies.parquet', compression='brotli')

# ---------------- Users Dataset ----------------

users_df_columns = [
    'user_id',
    'user_gender',
    # 'raw_user_age',
    'user_zip_code',
    'bucketized_user_age',
    # 'user_occupation_text',
    'user_occupation_label',
]
users_df = df.loc[:, users_df_columns]

users_df.rename(
    columns = {
        'bucketized_user_age': 'user_bucketized_age'
    },
    inplace = True
)

users_df['user_gender'] = users_df['user_gender'].apply(lambda x: int(x))  # Cast booleans to integers

# Remove duplicates
users_df.drop_duplicates(inplace=True)

users_df.to_parquet(f'data/{DATASET_SIZE}-users.parquet', compression='brotli')
