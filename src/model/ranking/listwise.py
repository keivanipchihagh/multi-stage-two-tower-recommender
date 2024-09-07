from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Third-party
from src.model.ranking.base import BaseRanking

class ListwiseRanking(BaseRanking):

    def __init__(
        self,
        query_tower: tf.keras.Model,
        candidate_tower: tf.keras.Model,
        task: tfrs.tasks.Ranking,
    ) -> 'ListwiseRanking':
        """
            Listwise Ranking Model.

            Parameters:
                - query_tower (tf.keras.Model): Query tower model.
                - candidate_tower (tf.keras.Model): Candidate tower model.
                - task (tfrs.tasks.Ranking): Ranking task for training.
        """
        super().__init__(query_tower, candidate_tower, task)


    def call(
        self,
        inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
            Call method of the model. Takes dict of input features and returns 
            predictions.

            Parameters:
                - inputs (Dict[str, tf.Tensor]): Dictionary of input Tensors.

            Returns:
                (tf.Tensor): Ranking scores.
        """
        query_embeddings: tf.Tensor     = self.query_tower(inputs)
        candidate_embeddings: tf.Tensor = self.candidate_tower(inputs)

        return self.rating_model(
            tf.concat(
                [
                    query_embeddings,
                    candidate_embeddings
                ], axis=-1
            )
        )


    def compute_loss(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False
    ) -> tf.Tensor:
        """
            Compute loss of the model.

            Parameters:
                - inputs (Dict[str, tf.Tensor]): Inputs of the model.
                - training (bool): If `True`, the model is in training mode.

            Returns:
                - (tf.Tensor): Loss of the model.
        """
        # Extracation
        labels: tf.Tensor      = inputs["user_rating"]
        predictions: tf.Tensor = self(inputs)

        return self.task(
            predictions     = tf.squeeze(predictions, axis=-1),
            labels          = labels,
            compute_metrics = not training
        )
