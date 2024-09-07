import numpy as np
from typing import List
import tensorflow as tf
from typing import Dict, List

class Embedding(tf.keras.Model):

    def __init__(
        self,
        dataset: tf.data.Dataset,
        embedding_dim: int,
        str_features: List[str] = [],
        int_features: List[str] = [],
        text_features: List[str] = [],
        timestamp_features: List[str] = [],
    ) -> 'Embedding':
        """
            Embedder Model.

            Parameters:
                - dataset (tf.data.Dataset): Values to make embeddings for.
                - embedding_dim (int): Embeddimg dimentionality.
                - str_features (List[str]): String features. Defaults to [].
                - int_features (List[str]): Integer features. Defaults to [].
                - text_features (List[str]): Textual features. Defaults to [].
                - timestamp_features (List[str]): Timestamp features. Defaults to [].
                - oov_token (str): OOV token. Defaults to "[UNK]".
        """
        super().__init__()

        # Embedding layers will be applied to each of the specified 
        # features, with all layers having the same dimensionality.
        self._embedding_dim = embedding_dim

        self.embeddings: Dict[str, tf.keras.Sequential] = {}

        # For string categorical features, the `StringLookup`
        # layer will create a vocabulary that maps each string
        # value to an integer index followed by an embedding layer.
        for feature in str_features:
            _embedding_layer = self.__create_str_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = self._embedding_dim,
            )
            self.embeddings[feature] = _embedding_layer

        # For integer categorical features, the `StringLookup`
        # layer will create a vocabulary that maps each string
        # value to an integer index followed by an embedding layer.
        for feature in int_features:
            _embedding_layer = self.__create_int_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = self._embedding_dim,
            )
            self.embeddings[feature] = _embedding_layer

        # For text features, the `TextVectorization` layer will
        # create a vocabulary that maps each token to an integer
        # index followed by an embedding layer.
        for feature in text_features:
            _embedding_layer = self.__create_text_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = self._embedding_dim,
            )
            self.embeddings[feature] = _embedding_layer

        # Timestamp features will be discretized into buckets
        # and the `Discretization` layer will create a vocabulary
        # for the embedding layer. The value will finally be
        # normalized between 0 and 1.
        for feature in timestamp_features:
            _embedding_layer = self.__create_timestamp_embedding_layer(
                values = dataset.map(lambda _: _[feature]),
                embedding_dim = self._embedding_dim,
            )
            self.embeddings[feature] = _embedding_layer


    def __create_text_embedding_layer(
        self,
        values: tf.data.Dataset,
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


    def __create_timestamp_embedding_layer(
        self,
        values: tf.data.Dataset,
        embedding_dim: int,
        n_buckets: int = 1000,
    ) -> tf.keras.Sequential:

        timestamps = np.concatenate(list(values))

        buckets = np.linspace(
            start = timestamps.min(),
            stop = timestamps.max(),
            num = n_buckets,
        )

        embedding_layer = tf.keras.layers.Embedding(
            # Size of the vocabulary
            input_dim = len(buckets) + 1,
            # Dimension of the dense embedding
            output_dim = embedding_dim
        )

        return tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(buckets.tolist()),
                embedding_layer,
                tf.keras.layers.Normalization(axis = None)
            ]
        )


    def __create_int_embedding_layer(
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
            Calls the model on inputs.  Takes dict of input features and returns 
            the concatenated embeddings.

            Parameters:
                - inputs (Dict[str, tf.Tensor]): Dictionary of Tensors.

            Returns:
                - (tf.tensor): returns the concatenated embeddings.
        """
        embeddings: List[tf.Tensor] = []
        for feature in self.embeddings.keys():
            embedding_layer = self.embeddings[feature]
            embedding = embedding_layer(inputs[feature])
            embeddings.append(embedding)

        return tf.concat(embeddings, axis=-1)


    @property
    def embeddings_output_dim(self) -> int:
        """
            The output dimension of the embedding layer.
        """
        return len(self.embeddings.keys()) * self._embedding_dim
