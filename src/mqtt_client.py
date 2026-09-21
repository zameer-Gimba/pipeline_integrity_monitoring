import json
import queue
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:  # Allows the core project to run without MQTT installed.
    mqtt = None


class MQTTClient:
    """Small MQTT publisher/subscriber wrapper for the prototype."""

    def __init__(self, broker, port=1883, topic="pipeline/prototype/telemetry"):
        if mqtt is None:
            raise RuntimeError("Install paho-mqtt to use MQTT support.")

        self.broker = broker
        self.port = port
        self.topic = topic
        self.messages = queue.Queue()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        self.client.on_message = self._on_message

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {"raw": message.payload.decode("utf-8", errors="replace")}

        self.messages.put(payload)

    def connect(self):
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.subscribe(self.topic)
        self.client.loop_start()

    def publish(self, payload):
        message = dict(payload)
        message.setdefault("timestamp", time.time())
        self.client.publish(self.topic, json.dumps(message), qos=1)

    def poll(self):
        items = []
        while True:
            try:
                items.append(self.messages.get_nowait())
            except queue.Empty:
                return items

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
