import tensorflow as tf

# Third-party
from src.models.movie import MovieEmbeddingModel


class CandidateTower(tf.keras.Model):

    def __init__(
        self,
        dataset,
        embedding_dim,
    ) -> 'CandidateTower':
        """
            Candidate Tower
        """
        super().__init__()

        self.embedding_model = MovieEmbeddingModel(
            dataset = dataset,
            embedding_dim = embedding_dim
        )


    def call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        embedding: tf.Tensor = self.embedding_model(inputs)
        return embedding
