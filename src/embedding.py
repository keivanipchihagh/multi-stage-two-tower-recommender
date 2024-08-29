import numpy as np
import tensorflow as tf
from typing import Dict, List

class EmbeddingModel(tf.keras.Model):

    def __init__(
        self,
        dataset: tf.data.Dataset,
        embedding_dim: int,
        str_features: List[str] = [],
        int_features: List[str] = [],
        text_features: List[str] = [],
    ) -> 'EmbeddingModel':
        """
            Embedder Model.

            Parameters:
                - dataset (tf.data.Dataset): Values to make embeddings for.
                - embedding_dim (int): Embeddimg dimentionality.
                - str_features (List[str]): String features. Defaults to [].
                - int_features (List[str]): Integer features. Defaults to [].
                - text_features (List[str]): Textual features. Defaults to [].
                - oov_token (str): OOV token. Defaults to "[UNK]".
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self._all_features = str_features + int_features + text_features

        self._embeddings: Dict[str, tf.keras.Sequential] = {}

        # String features
        for feature in str_features:
            _embedding_layer = self.__create_str_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = embedding_dim,
            )
            self._embeddings[feature] = _embedding_layer

        # Integer features
        for feature in int_features:
            _embedding_layer = self.__create_int_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = embedding_dim,
            )
            self._embeddings[feature] = _embedding_layer

        # Textual features
        for feature in text_features:
            _embedding_layer = self.__create_text_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = embedding_dim,
            )
            self._embeddings[feature] = _embedding_layer


    def __create_text_embedding_layer(
        self,
        values,
        embedding_dim: int
    ) -> tf.keras.Sequential:
        """
            Build a model that takes raw text values in and yields embeddings
            using a text vectorization.

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


    def __create_str_embedding_layer(
        self,
        values: tf.data.Dataset,
        embedding_dim: int,
    ) -> tf.keras.Sequential:
        """
            Build a model that takes raw string values in and yields embeddings
            using a table-based vocabulary string lookup.

            parameters:
                - values (tf.data.Dataset): Values to make embeddings for.
                - embedding_dim (int): Embeddimg dimentionality.
            Returns:
                (tf.keras.Sequential): Model.
        """
        vocabulary = np.unique(np.concatenate(list(values)))

        # Map arbitrary strings into integer output via a table-based vocabulary lookup.
        lookup_layer = tf.keras.layers.StringLookup(
            mask_token = None,
            # Vocabulary
            vocabulary = vocabulary,
        )

        embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim = len(vocabulary) + 1,
            # Dimension of the dense embedding
            output_dim = embedding_dim
        )

        return tf.keras.Sequential(
            [
                lookup_layer,
                embedding_layer
            ]
        )


    def __create_int_embedding_layer(
        self,
        values,
        embedding_dim: int,
    ) -> tf.keras.Sequential:
        """
            Build a model that takes raw string values in and yields embeddings
            using a table-based vocabulary string lookup.

            parameters:
                - values (tf.data.Dataset): Values to make embeddings for.
                - embedding_dim (int): Embeddimg dimentionality.
            Returns:
                (tf.keras.Sequential): Model.
        """
        vocabulary = np.unique(np.concatenate(list(values)))

        # Map arbitrary strings into integer output via a table-based vocabulary lookup.
        lookup_layer = tf.keras.layers.IntegerLookup(
            mask_token = None,
            # Vocabulary
            vocabulary = vocabulary,
        )

        embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim = len(vocabulary) + 1,
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
        embeddings: List[tf.Tensor] = []
        for feature in self._all_features:
            embedding_layer = self._embeddings[feature]
            embedding = embedding_layer(inputs[feature])
            embeddings.append(embedding)

        return tf.concat(embeddings, axis=1)
