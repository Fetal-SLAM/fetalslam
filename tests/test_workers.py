import threading
import time

from fetal_slam.workers import BackgroundWorker


def test_worker_allows_one_outstanding_job():
    started = threading.Event()
    release = threading.Event()

    def task(value):
        started.set()
        release.wait(timeout=2)
        return value * 2

    worker = BackgroundWorker("test-worker", task)
    worker.start()
    try:
        assert worker.submit(21)
        assert started.wait(timeout=1)
        assert worker.busy
        assert not worker.submit(22)

        release.set()
        results = []
        for _ in range(100):
            results = worker.poll()
            if results:
                break
            time.sleep(0.01)

        assert results[0].value == 42
        assert results[0].error is None
        assert not worker.busy
    finally:
        release.set()
        worker.stop()
