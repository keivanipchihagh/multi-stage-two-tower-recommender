from typing import Dict
import tensorflow as tf

# Third-party
from src.utils.timer import log_execution_time

class UserEmbeddingModel(tf.keras.Model):

    def __init__(
        self,
        dataset,
        embedding_dim: int,
        oov_token: str = "[UNK]",
    ) -> 'UserEmbeddingModel':
        """
            User Embedding Model
        """
        super().__init__()

        self.oov_token = oov_token

        self.id_embedding_layer = self.__create_id_embedding_layer(
            values = dataset.map(lambda _: _['user_id']),
            embedding_dim = embedding_dim,
        )


    @log_execution_time
    def __create_id_embedding_layer(
        self,
        values,
        embedding_dim: int
    ) -> tf.keras.Sequential:
        """
            Build a model that takes raw values in and yields embeddings.

            parameters:
                - values (tf.data.Dataset): Values to make embeddings for.
                - embedding_dim (int): Embeddimg dimentionality.
            Returns:
                (tf.keras.Sequential): Model.
        """

        # Map arbitrary strings into integer output via a table-based vocabulary lookup.
        lookup_layer = tf.keras.layers.StringLookup(
            mask_token = None,
            # Out of Vocabulary Token
            oov_token = self.oov_token,
        )

        # StringLookup layer is a non-trainable layer and its state (the vocabulary)
        # must be constructed and set before training in a step called "adaptation".
        lookup_layer.adapt(values)

        embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim = lookup_layer.vocabulary_size(),
            # Dimension of the dense embedding
            output_dim = embedding_dim
        )

        return tf.keras.Sequential(
            [
                lookup_layer,
                embedding_layer
            ]
        )


    def call(
        self,
        inputs: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """
            Calls the model on inputs.

            Parameters:
                - inputs (Dict[str, tf.Tensor]): Dictionary of Tensors.

            Returns:
                - (tf.tensor): returns the concatenated embeddings.
        """
        # Extracation
        id: tf.Tensor = inputs["user_id"]

        # Create embeddings
        id_embedding    = self.id_embedding_layer(id)

        return tf.concat(
            [
                id_embedding,
            ], axis=1
        )
