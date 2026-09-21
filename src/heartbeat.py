import time


class HeartbeatMonitor:
    """Tracks whether an IoT node has reported recently."""

    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.last_seen = None

    def receive(self, timestamp=None):
        self.last_seen = timestamp if timestamp is not None else time.time()

    def is_online(self, now=None):
        if self.last_seen is None:
            return False
        current = now if now is not None else time.time()
        return (current - self.last_seen) <= self.timeout_seconds

    def seconds_since_last_seen(self, now=None):
        if self.last_seen is None:
            return None
        current = now if now is not None else time.time()
        return max(0.0, current - self.last_seen)
