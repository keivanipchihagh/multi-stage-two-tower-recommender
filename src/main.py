import uvicorn
from prometheus_client import start_http_server

# Third-party
from config import (
    API_PORT,
    API_LOG_LEVEL,
    API_RELOAD,
    API_WORKERS,
    PROMETHEUS_SERVER_PORT,
)


if __name__ == "__main__":

    # Prometheus
    start_http_server(PROMETHEUS_SERVER_PORT)

    # Api
    uvicorn.run(
        app       = "api:APP",
        host      = "0.0.0.0",
        port      = API_PORT,
        reload    = API_RELOAD,
        workers   = API_WORKERS,
        log_level = API_LOG_LEVEL,
    )
