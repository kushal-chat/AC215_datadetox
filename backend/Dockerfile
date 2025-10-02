FROM python:3.11-slim-bookworm

ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    mkdir -p /app

WORKDIR /app
COPY --chown=app:app . ./

RUN uv sync

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "source /home/app/.venv/bin/activate && exec bash"]
