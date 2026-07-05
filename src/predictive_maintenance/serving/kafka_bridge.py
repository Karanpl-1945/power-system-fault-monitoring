from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("predictive_maintenance.kafka_bridge")


class ConnectionManager:
    """Tracks connected dashboard WebSocket clients and broadcasts to all of them."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        async with self._lock:
            targets = list(self._connections)
        for connection in targets:
            try:
                await connection.send_text(payload)
            except Exception:  # noqa: BLE001 - a broken client should not break the broadcast
                await self.disconnect(connection)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


class KafkaAlertBridge:
    """Background thread that relays live Kafka predictions to WebSocket clients.

    Runs independently of the inference consumer (`scripts/kafka_inference_consumer.py`):
    that process does the real model inference and publishes results to Kafka; this
    bridge only re-reads that output topic and fans it out to connected browsers.
    Kafka being down/unreachable must never take down the API - failures here are
    caught, logged, and retried with backoff instead of raised.
    """

    def __init__(
        self,
        manager: ConnectionManager,
        loop: asyncio.AbstractEventLoop,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
    ) -> None:
        self._manager = manager
        self._loop = loop
        self._bootstrap_servers = bootstrap_servers or os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._topic = topic or os.environ.get("PREDICTION_TOPIC", "power.fault.predictions")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "stopped"
        self.last_error: str | None = None
        self.messages_relayed = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kafka-alert-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:
            self.status = "unavailable"
            self.last_error = f"confluent_kafka not installed: {exc}"
            logger.warning(self.last_error)
            return

        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            self.status = "connecting"
            consumer = None
            try:
                consumer = Consumer(
                    {
                        "bootstrap.servers": self._bootstrap_servers,
                        "group.id": f"dashboard-bridge-{uuid.uuid4()}",
                        "auto.offset.reset": "latest",
                        "enable.auto.commit": False,
                    }
                )
                # subscribe() is lazy and never fails on its own, and poll() just
                # returns None on a connection timeout instead of raising - so without
                # this probe an unreachable broker would be misreported as "connected".
                consumer.list_topics(topic=self._topic, timeout=5.0)
                consumer.subscribe([self._topic])
                self.status = "connected"
                self.last_error = None
                backoff_seconds = 1.0

                while not self._stop_event.is_set():
                    message = consumer.poll(1.0)
                    if message is None:
                        continue
                    if message.error():
                        raise RuntimeError(str(message.error()))
                    try:
                        payload = json.loads(message.value())
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("kafka_bridge_bad_payload=%s", exc)
                        continue

                    self.messages_relayed += 1
                    asyncio.run_coroutine_threadsafe(
                        self._manager.broadcast(payload), self._loop
                    )
            except Exception as exc:  # noqa: BLE001 - keep the bridge alive on any Kafka error
                self.status = "unavailable"
                self.last_error = str(exc)
                logger.warning("kafka_bridge_error=%s", exc)
            finally:
                if consumer is not None:
                    consumer.close()

            if self._stop_event.wait(backoff_seconds):
                break
            backoff_seconds = min(backoff_seconds * 2, 30.0)

        self.status = "stopped"
