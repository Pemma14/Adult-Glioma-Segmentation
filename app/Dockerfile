FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uv/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN /uv/bin/uv pip install --system --no-cache -r /app/pyproject.toml

COPY . /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8500

CMD ["python", "-m", "app.main"]
