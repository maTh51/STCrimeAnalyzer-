# pull official base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#FROM nvcr.io/nvidia/pytorch:21.05-py3
#MAINTAINER Matheus Pereira <matheuspereira@dcc.ufmg.br>

# set working directory
WORKDIR /

# set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install kafkacat\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install --upgrade pip 
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "scripts/general.py"]