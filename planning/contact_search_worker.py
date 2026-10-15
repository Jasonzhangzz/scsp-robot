"""Single-flight background contact search with wall-clock failure backoff."""
from concurrent.futures import ThreadPoolExecutor
import time


class ContactSearchWorker:
    def __init__(self, *, clock=time.monotonic, retry_interval=.25, max_retry_interval=4.):
        self.clock = clock
        self.retry_interval = float(retry_interval)
        self.max_retry_interval = float(max_retry_interval)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lambda-regrasp')
        self.future = None
        self.next_attempt = -float('inf')
        self.failures = 0

    @property
    def ready(self):
        return self.future is None and self.clock() >= self.next_attempt

    @property
    def pending(self):
        return self.future is not None

    def submit(self, search):
        if not self.ready:
            return False
        self.future = self.executor.submit(search)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        try:
            value = future.result()
        except (FloatingPointError, RuntimeError) as error:
            self.failures += 1
            delay = min(self.max_retry_interval, self.retry_interval*2**min(self.failures-1, 8))
            self.next_attempt = self.clock()+delay
            return dict(error=str(error), retry_after=delay)
        self.failures = 0
        self.next_attempt = self.clock()+self.retry_interval
        return dict(result=value)

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
