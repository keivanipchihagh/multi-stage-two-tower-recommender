import tensorflow as tf
from typing import Dict, Tuple
import tensorflow_recommenders as tfrs

# Third-party
from src.model.retrieval import Retrieval
from src.model.ranking.base import BaseRanking

class RecommenderModel(tfrs.models.Model):

    def __init__(
        self,
        query_tower: tf.keras.Model,
        candidate_tower: tf.keras.Model,
        ranking_model: BaseRanking,
        retrieval_model: Retrieval,
        ranking_weight: float = 1.0,
        retrieval_weight: float = 1.0,
    ) -> 'RecommenderModel':
        """
            Initializes a RecommenderModel.

            Parameters:
                - query_tower (tf.keras.Model): The query tower of the model, which takes in queries and outputs a dense embedding.
                - candidate_tower (tf.keras.Model): The candidate tower of the model, which takes in candidates and outputs a dense embedding.
                - ranking_model (BaseRanking): The ranking task of the model. This task takes in the embeddings from the query and candidate towers and outputs a score for each candidate.
                - Retrieval_model (Retrieval): The retrieval task of the model. This task takes in the embeddings from the query and candidate towers and outputs a list of candidates.
                - ranking_weight (float): The weight given to the ranking task. Defaults to 1.0.
                - retrieval_weight (float): The weight given to the retrieval task. Defaults to 1.0.
        """
        super().__init__()

        self.query_tower      = query_tower
        self.candidate_tower  = candidate_tower
        self.ranking_weight   = ranking_weight
        self.retrieval_weight = retrieval_weight
        self.ranking_model    = ranking_model
        self.retrieval_model  = retrieval_model


    def call(
        self,
        inputs: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
            Call method of the model. Takes dict of inputs and returns predictions.

            Parameters:
                - inputs (Dict[str, tf.Tensor]): Dictionary of input Tensors.

            Returns:
                - (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
                    - query_embeddings (tf.Tensor): Query embeddings.
                    - candidate_embeddings (tf.Tensor): Candidate embeddings.
                    - ratings (tf.Tensor): Ratings.
        """
        ratings: tf.Tensor = self.ranking_model(inputs)
        return ratings


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
        rating_loss: tf.Tensor    = self.ranking_model.compute_loss(inputs, training)
        retrieval_loss: tf.Tensor = self.retrieval_model.compute_loss(inputs, training)

        ranking_weighted_loss   = (self.ranking_weight * rating_loss)
        retrieval_weighted_loss = (self.retrieval_weight * retrieval_loss)

        return ranking_weighted_loss + retrieval_weighted_loss
