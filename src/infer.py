from typing import Dict, Any

import scann
import tensorflow as tf

from config import (
    SCANN_PATH,
    BRUTE_PATH
)

scann_retrieval = tf.saved_model.load(SCANN_PATH)
brute_retrieval = tf.saved_model.load(BRUTE_PATH)


def retrieve(
    user: Dict[str, Any],
    k: int,
    approximate: bool = True
) -> list:
    """
        Perform retrieval for a given user.

        Parameters:
            - user (Dict[str, Any]): A dictionary containing the user's features.
            - k (int): The number of items to retrieve.
            - approximate (bool): Whether to use an approximate nearest neighbors 
                search or an exact search. Defaults to `True`.

        Returns:
            - identifiers (list): A list of item identifiers.
    """
    user_tensor = {k: tf.convert_to_tensor([v]) for k, v in user.items()}

    if approximate:
        _ = scann_retrieval.signatures['call'](**user_tensor, k=k)  # Approximate
    else:
        _ = brute_retrieval.signatures['call'](**user_tensor, k=k)  # Exact

    identifiers = _['output_0'].numpy().tolist()
    affnities   = _['output_1'].numpy().tolist()

    return identifiers
