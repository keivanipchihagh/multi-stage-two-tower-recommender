from typing import Dict
import tensorflow as tf


class MovieEmbeddingModel(tf.keras.Model):

    def __init__(
        self,
        dataset,
        embedding_dim: int,
        oov_token: str = "[UNK]",
    ) -> 'MovieEmbeddingModel':
        """
            Movie Embedding Model
        """
        super().__init__()

        self.oov_token = oov_token

        self.id_embedding_layer = self.__create_id_embedding_layer(
            values = dataset.map(lambda _: _['movie_id']),
            embedding_dim = embedding_dim,
        )
        self.title_embedding_layer = self.__create_title_embedding_layer(
            values = dataset.map(lambda _: _['movie_title']),
            embedding_dim = embedding_dim,
        )


    def __create_title_embedding_layer(
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

        # Transform a batch of strings into either a list of token indices.
        vectorization_layer = tf.keras.layers.TextVectorization()
        vectorization_layer.adapt(values)

        embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim = vectorization_layer.vocabulary_size(),
            # Dimension of the dense embedding
            output_dim = embedding_dim,
            # Whether or not the input value 0 is a special "padding" value that should be masked out.
            mask_zero = True
        )

        return tf.keras.Sequential(
            [
                vectorization_layer,
                embedding_layer,
                # Each title contains multiple words, so we will get multiple embeddings
                # for each title that should be compressed into a single embedding for
                # the text. Models like RNNs, Transformers or Attentions are useful here.
                # However, averaging all the words' embeddings together is also a good
                # starting point.
                tf.keras.layers.GlobalAveragePooling1D()
            ]
        )


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
        id: tf.Tensor    = inputs["movie_id"]
        title: tf.Tensor = inputs["movie_title"]

        # Create embeddings
        id_embedding: tf.Tensor    = self.id_embedding_layer(id)
        title_embedding: tf.Tensor = self.title_embedding_layer(title)

        return tf.concat(
            [
                id_embedding,
                title_embedding
            ], axis=1
        )
