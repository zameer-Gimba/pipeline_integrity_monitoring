class TamperMonitor:
    """Represents the physical tamper signal from a protected monitoring node."""

    def __init__(self):
        self.tamper_detected = False

    def set_state(self, detected: bool):
        self.tamper_detected = bool(detected)

    @property
    def status(self):
        return "TAMPER ALERT" if self.tamper_detected else "SECURE"
