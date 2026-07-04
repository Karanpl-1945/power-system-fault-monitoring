FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "predictive_maintenance.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

