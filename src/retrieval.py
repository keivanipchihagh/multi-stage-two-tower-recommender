from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs


class RetrievalModel(tfrs.models.Model):
 
    def __init__(
        self,
        query_tower: tf.keras.Model,
        candidate_tower: tf.keras.Model,
        task: tf.keras.layers.Layer,
    ) -> 'RetrievalModel':
        """
            Retrieval Model.

            Parameters:
                - query_tower (tf.keras.Model): Model for query tower.
                - candidate_tower (tf.keras.Model): Model for candidate tower.
                - task (tfrs.tasks.Retrieval): Retrieval task for training.
        """
        super().__init__()

        self.query_tower     = query_tower
        self.candidate_tower = candidate_tower
        self.task            = task


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
        query_embeddings: tf.Tensor     = self.query_tower(inputs)
        candidate_embeddings: tf.Tensor = self.candidate_tower(inputs)


        loss = self.task(
            query_embeddings     = query_embeddings,
            candidate_embeddings = candidate_embeddings,
            compute_metrics      = not training  # Speed up training
        )
        return loss
