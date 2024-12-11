import logging
import brotli
import random
import numpy as np
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from fastapi import FastAPI, APIRouter, Query, Body
from config import get_settings
import asyncio
from aiokafka.errors import KafkaConnectionError
import json
import time

from data_handler import read_df_offline, pre_process
from grid import create_grid, plot_grid
from class_st3dnet import ST3DNETModel

log = logging.getLogger("uvicorn")

# Producer/Consumer Application Setup
router = APIRouter(prefix="/st3dnet")


# Compress the message
async def compress(message: str) -> bytes:
    return brotli.compress(
        bytes(message, get_settings().file_encoding),
        quality=get_settings().file_compression_quality,
    )


async def compress_json(message: str) -> bytes:
    return brotli.compress(message.encode("utf-8"))


# Decompress the message
async def decompress(file_bytes: bytes) -> str:
    return str(
        brotli.decompress(file_bytes),
        get_settings().file_encoding,
    )


async def decompress_json(file_bytes: bytes) -> str:
    return brotli.decompress(file_bytes).decode("utf-8")


# Route to produce a message to the Kafka topic
@router.post("/")
async def produce_message(message: dict = Body(...)) -> dict:

    message_json = json.dumps(message)

    metadata = await producer.send_and_wait(
        "model_request",
        await compress_json(message_json),
        headers=[("target_service", b"st3dnet")],
    )

    return {"metadata": metadata}


# Function to consume messages from the Kafka topic
async def consume_reqs(consumer):
    await consumer.start()
    try:
        async for msg in consumer:
            headers = dict(msg.headers)
            if headers.get("target_service") == b"st3dnet":
                log.info("Message Consumed:")

                config = await decompress_json(msg.value)
                config = json.loads(config)

                df = read_df_offline(config, "path_substituido")
                points = pre_process(
                    data=df,
                    neighborhood=config["database"]["filters"]["nome_municipio"],
                    columns=config["database"]["columns"],
                )

                grid = create_grid(
                    config["evaluation"]["grid_size"],
                    config["database"]["filters"]["nome_municipio"],
                )

                train_points = points.query(
                    f"data_hora_fato <= '{config['evaluation']['test_end_date']}'"
                ).copy()

                test_points = points.query(
                    f"'{config['evaluation']['train_end_date']}' \
                                            < data_hora_fato <= \
                                            '{config['evaluation']['test_end_date']}'"
                ).copy()

                len_closeness, len_period, len_trend = 6, 0, 4
                st3dnet = ST3DNETModel(
                    train_points,
                    grid,
                    config["evaluation"]["train_end_date"],
                    len_closeness,
                    len_period,
                    len_trend,
                )
                st3dnet.train()

                predicts = st3dnet.predict(config["evaluation"]["test_end_date"])
                log.info(", ".join(str(x) for x in predicts.tolist()))

                await producer.send_and_wait(
                    "model_response",
                    await compress(json.dumps(predicts.tolist())),
                    headers=[("target_service", b"st3dnet")],
                )
            else:
                print("Message not for this service, skipping...")
                log.info("Message not for this service, skipping...")
    finally:
        await consumer.stop()


# Create the FastAPI application
def create_application() -> FastAPI:
    application = FastAPI(openapi_url="/st3dnet/openapi.json", docs_url="/st3dnet/docs")
    application.include_router(router, tags=["producer"])
    return application


# Create Kafka Producer
def create_producer() -> AIOKafkaProducer:
    return AIOKafkaProducer(
        bootstrap_servers=get_settings().kafka_instance,
    )


# Create Kafka Consumer
def create_consumer() -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        "model_request",  # Specific topic
        bootstrap_servers=get_settings().kafka_instance,
        # group_id="my_consumer_group",  # Group ID to ensure only one service processes the message
    )


# Initialize FastAPI application and Kafka Producer/Consumer
async def wait_for_kafka():
    while not await check_kafka():
        log.info("Kafka não disponível ainda. Aguardando...")
        await asyncio.sleep(5)
    log.info("Kafka está disponível.")


async def check_kafka():
    producer = AIOKafkaProducer(bootstrap_servers="kafka:9092")
    try:
        await producer.start()
        await producer.stop()
        return True
    except KafkaConnectionError:
        return False


async def wait_for_kafka():
    while not await check_kafka():
        log.info("Kafka não disponível ainda. Aguardando...")
        await asyncio.sleep(5)
    log.info("Kafka está disponível.")


app = create_application()
producer = create_producer()
consumer = create_consumer()


@app.on_event("startup")
async def startup_event():
    """Start up event for FastAPI application."""
    log.info("Aguardando Kafka ficar disponível...")
    await wait_for_kafka()  # Espera o Kafka estar disponível
    log.info("Starting up...")
    await producer.start()
    asyncio.create_task(consume_reqs(consumer))


@app.on_event("shutdown")
async def shutdown_event():
    log.info("Shutting down...")
    await producer.stop()
    await consumer.stop()
