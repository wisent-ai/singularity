FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    pydantic

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
