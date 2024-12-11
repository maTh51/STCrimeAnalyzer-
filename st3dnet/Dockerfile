# pull official base image
FROM tensorflow/tensorflow:2.16.1-gpu

# set working directory
WORKDIR /

# set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install build-essential libpq-dev python3-dev \
    && apt-get -y install kafkacat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install --upgrade pip 
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# add entrypoint.sh
# COPY ./entrypoint.sh .
# RUN chmod +x entrypoint.sh

COPY . .
# run entrypoint.sh
ENTRYPOINT ["sh", "-c","uvicorn main:app --host=0.0.0.0 --port=8001  --reload"]
