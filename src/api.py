from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Request, status
from prometheus_fastapi_instrumentator import Instrumentator

# Third-party
from infer import retrieve, rank


APP = FastAPI()
Instrumentator().instrument(APP).expose(APP)


class UserModel(BaseModel):
    user_id: str
    user_gender: int
    user_zip_code: str
    user_bucketized_age: float
    user_occupation_label: int

class MovieModel(BaseModel):
    movie_id: str
    movie_title: str
    movie_release_year: str


@APP.get(
    path = "/api/healthcheck",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'healthcheck'],
)
async def healthcheck(request: Request):
    return "OK"


@APP.get(
    path = "/api/v1/retrieval",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'v1', 'retrieval'],
)
async def api_v1_retrieval(
    user: UserModel,
    top_k: int = 10,
    approximate: bool = True
):
    data = user.model_dump()
    return retrieve(
        user = data,
        k = top_k,
        approximate = approximate,
    )


@APP.get(
    path = "/api/v1/ranking",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'v1', 'ranking'],
)
async def api_v1_rank(movies: List[MovieModel], user: UserModel):

    user_dict = user.model_dump()

    movie_scores = {}
    for movie in movies:
        movie_dict = movie.model_dump()
        movie_id = movie_dict.get('movie_id')

        score = rank(user_dict, movie_dict)
        
        movie_scores[movie_id] = score

    return movie_scores
