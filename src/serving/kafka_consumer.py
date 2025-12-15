"""Kafka Consumer for Fraud Detection Service.

This module provides a Kafka consumer that listens to the
fraud-evaluation-request topic and publishes results to
fraud-evaluation-result topic.

Note: This is a reference implementation. Actual Kafka configuration
should be provided via environment variables in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from schemas.kafka_schemas import FraudEvaluationRequest, FraudEvaluationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fraud-detection-consumer")

# Kafka configuration from environment
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = os.getenv("KAFKA_REQUEST_TOPIC", "fraud-evaluation-request")
RESULT_TOPIC = os.getenv("KAFKA_RESULT_TOPIC", "fraud-evaluation-result")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-detection-service")


class FraudDetectionKafkaService:
    """Kafka-based fraud detection service.
    
    Consumes transaction events from fraud-evaluation-request topic,
    evaluates fraud risk, and produces results to fraud-evaluation-result topic.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        request_topic: str = REQUEST_TOPIC,
        result_topic: str = RESULT_TOPIC,
        consumer_group: str = CONSUMER_GROUP,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.request_topic = request_topic
        self.result_topic = result_topic
        self.consumer_group = consumer_group
        self._running = False
        self._evaluator = None
        
    def _get_evaluator(self):
        """Lazy-load the fraud evaluator."""
        if self._evaluator is None:
            from serving.fraud_evaluator import FraudEvaluator
            self._evaluator = FraudEvaluator()
        return self._evaluator
    
    async def process_message(self, message_value: bytes) -> Optional[str]:
        """Process a single Kafka message.
        
        Parameters
        ----------
        message_value : bytes
            Raw message value from Kafka.
        
        Returns
        -------
        Optional[str]
            JSON string of FraudEvaluationResult, or None on error.
        """
        try:
            # Parse message
            data = json.loads(message_value.decode("utf-8"))
            request = FraudEvaluationRequest(**data)
            
            logger.info(f"Processing transaction: {request.transaction_id}")
            
            # Evaluate fraud risk
            evaluator = self._get_evaluator()
            result = evaluator.evaluate_transaction(request)
            
            logger.info(
                f"Transaction {request.transaction_id}: "
                f"score={result.risk_score:.4f}, "
                f"level={result.risk_level}, "
                f"action={result.recommended_action}"
            )
            
            # Convert to JSON for Kafka
            return result.model_dump_json(by_alias=True)
            
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON message: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Error processing message: {exc}")
            return None
    
    async def run_with_aiokafka(self):
        """Run the consumer using aiokafka library.
        
        This is the preferred async implementation for production.
        Requires: pip install aiokafka
        """
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError:
            logger.error(
                "aiokafka not installed. Run: pip install aiokafka"
            )
            return
        
        consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
        )
        
        await consumer.start()
        await producer.start()
        
        logger.info(f"Started consuming from {self.request_topic}")
        logger.info(f"Publishing results to {self.result_topic}")
        
        self._running = True
        
        try:
            async for message in consumer:
                if not self._running:
                    break
                    
                result_json = await self.process_message(message.value)
                
                if result_json:
                    await producer.send_and_wait(
                        self.result_topic,
                        result_json.encode("utf-8")
                    )
                    
        finally:
            await consumer.stop()
            await producer.stop()
            logger.info("Consumer stopped")
    
    def run_with_confluent_kafka(self):
        """Run the consumer using confluent-kafka library.
        
        This is a synchronous implementation using confluent-kafka.
        Requires: pip install confluent-kafka
        """
        try:
            from confluent_kafka import Consumer, Producer, KafkaError
        except ImportError:
            logger.error(
                "confluent-kafka not installed. Run: pip install confluent-kafka"
            )
            return
        
        consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.consumer_group,
            "auto.offset.reset": "earliest",
        }
        
        producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
        }
        
        consumer = Consumer(consumer_config)
        producer = Producer(producer_config)
        
        consumer.subscribe([self.request_topic])
        
        logger.info(f"Started consuming from {self.request_topic}")
        logger.info(f"Publishing results to {self.result_topic}")
        
        self._running = True
        
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Message delivery failed: {err}")
            else:
                logger.debug(f"Message delivered to {msg.topic()}")
        
        try:
            while self._running:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Process synchronously (wrap in event loop if needed)
                loop = asyncio.new_event_loop()
                result_json = loop.run_until_complete(
                    self.process_message(msg.value())
                )
                loop.close()
                
                if result_json:
                    producer.produce(
                        self.result_topic,
                        result_json.encode("utf-8"),
                        callback=delivery_callback
                    )
                    producer.poll(0)
                    
        finally:
            consumer.close()
            producer.flush()
            logger.info("Consumer stopped")
    
    def stop(self):
        """Signal the consumer to stop."""
        self._running = False


def main():
    """Entry point for Kafka consumer service."""
    service = FraudDetectionKafkaService()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Try async first, fall back to sync
    try:
        import aiokafka
        logger.info("Using aiokafka (async)")
        asyncio.run(service.run_with_aiokafka())
    except ImportError:
        try:
            import confluent_kafka
            logger.info("Using confluent-kafka (sync)")
            service.run_with_confluent_kafka()
        except ImportError:
            logger.error(
                "No Kafka client installed. Install one of:\n"
                "  pip install aiokafka\n"
                "  pip install confluent-kafka"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
