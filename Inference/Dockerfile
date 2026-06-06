FROM ultralytics/ultralytics:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models

COPY app.py requirements.txt ./
COPY ./templates ./templates

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
