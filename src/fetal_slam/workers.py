import queue
import threading
from dataclasses import dataclass
from time import perf_counter


@dataclass
class WorkerResult:
    value: object
    elapsed: float
    error: Exception = None


class BackgroundWorker:
    """Single-consumer worker with bounded work and result queues."""

    def __init__(self, name, task):
        self.name = name
        self._task = task
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._stop_token = object()
        self._busy = threading.Event()
        self._scheduled = threading.Event()
        self._submit_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.jobs_completed = 0
        self.total_elapsed = 0.0

    def start(self):
        self._thread.start()

    def submit(self, value=None):
        with self._submit_lock:
            if self._scheduled.is_set():
                return False
            self._scheduled.set()
            try:
                self._requests.put_nowait(value)
                return True
            except queue.Full:
                self._scheduled.clear()
                return False

    @property
    def busy(self):
        return self._scheduled.is_set()

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                if results:
                    self._scheduled.clear()
                return results

    def stop(self):
        if not self._thread.is_alive():
            return
        self._requests.put(self._stop_token)
        self._thread.join()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is self._stop_token:
                self._requests.task_done()
                return

            self._busy.set()
            started = perf_counter()
            try:
                value = self._task(request)
                result = WorkerResult(value=value, elapsed=perf_counter() - started)
            except Exception as error:
                result = WorkerResult(
                    value=None,
                    elapsed=perf_counter() - started,
                    error=error,
                )
            finally:
                self._busy.clear()
                self._requests.task_done()

            self.jobs_completed += 1
            self.total_elapsed += result.elapsed
            self._results.put(result)
