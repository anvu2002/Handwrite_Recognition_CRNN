# FROM pytorch/pytorch:latest
# FROM  quay.io/jupyter/tensorflow-notebook:cuda-latest
FROM mher/flower
USER root
WORKDIR /app

# COPY requirements.txt requirements.txt
# RUN pip install -U pip
# RUN pip install -r requirements.txt
COPY celery_tasks celery_tasks
 