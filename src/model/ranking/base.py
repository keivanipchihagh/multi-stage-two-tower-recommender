from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs


class BaseRanking(tfrs.models.Model):

    def __init__(
        self,
        query_tower: tf.keras.Model,
        candidate_tower: tf.keras.Model,
        task: tfrs.tasks.Ranking,
    ) -> 'BaseRanking':
        """
            Ranking base Model.

            Parameters:
                - query_tower (tf.keras.Model): Query tower model.
                - candidate_tower (tf.keras.Model): Candidate tower model.
                - task (tfrs.tasks.Ranking): Ranking task for training.
        """
        super().__init__()

        self.query_tower     = query_tower
        self.candidate_tower = candidate_tower
        self.task            = task

        self.rating_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ]
        )


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
        raise NotImplementedError()


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
        raise NotImplementedError()