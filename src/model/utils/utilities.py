import math
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Text

plt.style.use('seaborn-v0_8')


def train_test_split(
    dataset: tf.data.Dataset,
    train_size: float,
    buffer_size: int = tf.data.UNKNOWN_CARDINALITY,
    random_state: int = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
        Split a dataset into train and test sets.

        Parameters:
            - dataset (tf.data.Dataset): The dataset to be split.
            - train_size (float): The proportion of the dataset to include in the train split.
            - buffer_size (int): The number of elements from which to sample the buffer.
            - random_state (int): The random seed used to create the distribution.

        Returns:
            - Tuple(tf.data.Dataset, tf.data.Dataset): The train and test datasets.
    """
    # Shuffle the elements of the dataset randomly.
    dataset_shuffled = dataset.shuffle(
        # The new dataset will be sampled from a buffer window of first `buffer_size`
        # elements of the dataset
        buffer_size = buffer_size,
        # The random seed used to create the distribution.
        seed = random_state,
        # Controls whether the shuffle order should be different for each epoch.
        # Can lead to data leakage (Use with causation).
        reshuffle_each_iteration = False
    )

    # Split dataset randomly
    trainset_size: int = train_size * dataset.__len__().numpy()

    train_dataset = dataset_shuffled.take(trainset_size)
    test_dataset = dataset_shuffled.skip(trainset_size)

    return train_dataset, test_dataset


def plot_history(
    history: tf.keras.callbacks.History,
    figsize: Tuple[int, int] = (20, 7),
    plot_training: bool = True
) -> None:
    """
        Plot the training and validation loss and accuracy of a model.
        
        Parameters:
            - history (tf.keras.callbacks.History): The History object returned by the fit method.
            - figsize (tuple[int, int]): The size of the figure. Defaults to (20, 7).
            - plot_training (bool): Whether to plot the training loss and accuracy. Defaults to True.
        
        Returns:
            None
    """
    loss_names = [key for key in history.history.keys() if 'val_' not in key]

    # Calculate the number of columns and rows
    num_losses = len(loss_names)
    num_cols   = math.ceil(num_losses / 3)
    num_rows   = 3

    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    for i, loss_name in enumerate(loss_names):
        loss: List[float]     = history.history.get(loss_name)
        loss_val: List[float] = history.history.get(f"val_{loss_name}", [0.0])

        if plot_training:
            axes[i].plot(loss, label='Training')
        axes[i].plot(loss_val, label='Validation')
        axes[i].set_title(f'{loss_name}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def _create_feature_dict(features: List[Text]) -> Dict[Text, List[tf.Tensor]]:
    """Helper function for creating an empty feature dict for defaultdict."""
    return {key: [] for key in features}


def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function for sampling a list example from given feature lists."""
    if random_state is None:
        random_state = np.random.RandomState()

    sampled_indices = random_state.choice(
        range(len(feature_lists["user_rating"])),
        size=num_examples_per_list,
        replace=False,
    )

    sampled_features = {}
    for name, values in feature_lists.items():
        sampled_features[name] = [
            values[idx] for idx in sampled_indices
        ]

    return {
        name: tf.stack(values, 0)
        for name, values in sampled_features.items()
    }


def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Function for converting the MovieLens 100K dataset to a listwise dataset.

    Args:
        rating_dataset:
        The MovieLens ratings dataset loaded from TFDS. Feature must be  provided
        in the dataset. The dataset must contain the "user_rating" feature.
        num_list_per_user:
        An integer representing the number of lists that should be sampled for
        each user in the training dataset.
        num_examples_per_list:
        An integer representing the number of movies to be sampled for each list
        from the list of movies rated by the user.
        seed:
        An integer for creating `np.random.RandomState`.

    Returns:
        A tf.data.Dataset containing list examples.

        Each example contains multiple keys. "user_id" maps to a string 
        tensor that represents the user id for the example. "movie_title" maps 
        to a tensor of shape [sum(num_example_per_list)] with dtype tf.string. 
        It represents the list of candidate movie ids. "user_rating" maps to 
        a tensor of shape [sum(num_example_per_list)] with dtype tf.float32. 
        It represents the rating of each movie in the candidate list.
    """
    random_state = np.random.RandomState(seed)

    features = rating_dataset.take(1).get_single_element().keys()
    example_lists_by_user = collections.defaultdict(lambda: _create_feature_dict(features))

    for example in rating_dataset:
        user_id = example.get('user_id').numpy()
        for key, value in example.items():
            example_lists_by_user[user_id][key].append(value.numpy())

    tensor_slices = {key: [] for key in features}

    for user_id, feature_lists in example_lists_by_user.items():
        for _ in range(num_list_per_user):

            # Drop the user if they don't have enough ratings.
            if len(feature_lists["user_rating"]) < num_examples_per_list:
                continue

            sampled_features = _sample_list(
                feature_lists,
                num_examples_per_list,
                random_state=random_state,
            )

            for feature, samples in sampled_features.items():
                tensor_slices[feature].append(samples)

    return tf.data.Dataset.from_tensor_slices(tensor_slices)
