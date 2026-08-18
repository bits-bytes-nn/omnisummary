FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the EXACT dependency set uv.lock pins — the same one CI resolves and the test suite
# exercises. `pip install .` re-resolved pyproject's version ranges at build time, so the image
# could ship a dependency set no test had ever run against. Dependencies are installed BEFORE the
# source is copied so a code-only change reuses this layer.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir --upgrade pip uv==0.6.14 && \
    uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY agent/ ./agent/
COPY agent_runtime/ ./agent_runtime/
COPY shared/ ./shared/
COPY pipeline/ ./pipeline/
COPY collectors/ ./collectors/
COPY output/ ./output/
COPY config/ ./config/
COPY lambda_handlers/ ./lambda_handlers/
COPY main.py ./

# --no-deps: the pinned set above is authoritative; installing the project must not pull anything
# newer. awslambdaric is the Lambda runtime shim, not an application dependency.
RUN uv pip install --system --no-cache --no-deps . && \
    pip install --no-cache-dir awslambdaric==4.0.2 && \
    rm -rf build omnisummary.egg-info

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_handlers.digest_handler.handler"]
