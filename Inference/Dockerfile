# FROM ultralytics/ultralytics:latest-jetson-jetpack6
# FROM ultralytics/ultralytics:latest
FROM python:3.13-slim

WORKDIR /app

# forse questo RUN non serve davvero
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY app.py requirements.txt ./

COPY ./templates ./templates
COPY ./models /models

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
