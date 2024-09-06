from pydantic import BaseModel
from fastapi import FastAPI, Request, status
from prometheus_fastapi_instrumentator import Instrumentator

# Third-party
from infer import retrieve

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
    movie_release_year: int


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
async def api_v1_rank(request: Request):
    raise NotImplementedError()
