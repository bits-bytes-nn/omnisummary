FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY agent/ ./agent/
COPY agent_runtime/ ./agent_runtime/
COPY shared/ ./shared/
COPY tools/ ./tools/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "agent_runtime.app"]
