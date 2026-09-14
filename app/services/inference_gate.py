"""Serialize CUDA work, allowing waiting searches ahead of gallery images."""
from contextlib import contextmanager
import threading


class InferenceGate:
    def __init__(self):
        self.condition = threading.Condition()
        self.active = False
        self.searches = 0

    @contextmanager
    def acquire(self, background=False):
        with self.condition:
            if not background:
                self.searches += 1
                self.condition.notify_all()
            try:
                while self.active or (background and self.searches):
                    self.condition.wait()
                self.active = True
            finally:
                if not background:
                    self.searches -= 1
        try:
            yield
        finally:
            with self.condition:
                self.active = False
                self.condition.notify_all()
