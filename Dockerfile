FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl postgresql-client && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
