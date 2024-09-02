import tensorflow as tf
from typing import Dict, Tuple
import tensorflow_recommenders as tfrs

# Third-party
from src.retrieval import Retrieval
from src.rankings.pointwise import PointwiseRanking


class RecommenderModel(tfrs.models.Model):
 
    def __init__(
        self,
        query_tower: tf.keras.Model,
        candidate_tower: tf.keras.Model,
        ranking_task: tfrs.tasks.Ranking,
        retrieval_task: tfrs.tasks.Retrieval,
        ranking_weight: float = 1.0,
        retrieval_weight: float = 1.0,
    ) -> 'RecommenderModel':
        """
            Initializes a RecommenderModel.

            Parameters:
                - query_tower (tf.keras.Model): A model for computing query embeddings.
                - candidate_tower (tf.keras.Model): A model for computing candidate embeddings.
                - ranking_task (tfrs.tasks.Ranking): A task for ranking.
                - retrieval_task (tfrs.tasks.Retrieval): A task for retrieval.
                - ranking_weight (float, optional): The weight for ranking loss. Defaults to 1.0.
                - retrieval_weight (float, optional): The weight for retrieval loss. Defaults to 1.0.
        """
        super().__init__()

        self.query_tower      = query_tower
        self.candidate_tower  = candidate_tower
        self.ranking_weight   = ranking_weight
        self.retrieval_weight = retrieval_weight

        self.ranking_model   = self.__create_ranking_model(ranking_task)
        self.retrieval_model = self.__create_retrieval_model(retrieval_task)


    def __create_ranking_model(
        self,
        task: tfrs.tasks.Ranking
    ) -> PointwiseRanking:
        """
            Create a ranking model.

            Parameters:
                - (tfrs.tasks.Ranking): A model for ranking.
        """
        return PointwiseRanking(
            query_tower     = self.query_tower,
            candidate_tower = self.candidate_tower,
            task            = task
        )


    def __create_retrieval_model(
        self,
        task: tfrs.tasks.Retrieval
    ) -> Retrieval:
        """
            Create a retrieval model.

            Parameters:
                - task (tfrs.tasks.Retrieval): A task for retrieval.

            Returns:
                - (Retrieval): A model for retrieval.
        """
        return Retrieval(
            query_tower     = self.query_tower,
            candidate_tower = self.candidate_tower,
            task            = task
        )


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
        query_embeddings: tf.Tensor     = self.query_tower(inputs)
        candidate_embeddings: tf.Tensor = self.candidate_tower(inputs)
        ratings: tf.Tensor              = self.ranking_model(inputs)

        return (
            query_embeddings,
            candidate_embeddings,
            ratings,
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
        rating_loss: tf.Tensor    = self.ranking_model.compute_loss(inputs, training)
        retrieval_loss: tf.Tensor = self.retrieval_model.compute_loss(inputs, training)

        ranking_weighted_loss   = (self.ranking_weight * rating_loss)
        retrieval_weighted_loss = (self.retrieval_weight * retrieval_loss)

        return ranking_weighted_loss + retrieval_weighted_loss
