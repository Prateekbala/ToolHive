FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[pipeline,router]" \
    && pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "httpx>=0.27.0"

COPY . .
RUN adduser --disabled-password --gecos "" toolhive \
    && chown -R toolhive:toolhive /app
USER toolhive
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
