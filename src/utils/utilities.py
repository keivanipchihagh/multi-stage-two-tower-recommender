import math
import pandas as pd
import tensorflow as tf
from typing import List, Tuple
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')


def dataframe_to_Dataset(
    df: pd.DataFrame,
    columns: List[str]
) -> tf.data.Dataset:
    """
        Convert a pandas DataFrame into a TensorFlow Dataset.

        Parameters:
        - df (pd.DataFrame): The pandas DataFrame to be converted.
        - columns (List[str]): The list of column names to be converted into a Dataset.

        Returns:
            - (tf.data.Dataset): The TensorFlow Dataset created from the DataFrame.
    """
    return tf.data.Dataset.from_tensor_slices(dict(df[columns]))


def train_test_split(
    dataset: tf.data.Dataset,
    train_size: float,
    buffer_size: int = tf.data.UNKNOWN_CARDINALITY,
    random_state: int = None,
) -> tf.data.Dataset:
    """
        Split a dataset into train and test sets.

        Parameters:
            - dataset (tf.data.Dataset): The dataset to be split.
            - train_size (float): The proportion of the dataset to include in the train split.
            - buffer_size (int): The number of elements from which to sample the buffer.
            - random_state (int): The random seed used to create the distribution.

        Returns:
            - (tf.data.Dataset, tf.data.Dataset): The train and test datasets.
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
