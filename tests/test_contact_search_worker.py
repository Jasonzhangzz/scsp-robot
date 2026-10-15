from threading import Event
from concurrent.futures import wait

from planning.contact_search_worker import ContactSearchWorker


def test_slow_failed_search_does_not_block_control_or_enqueue_repeated_jobs():
    entered, release = Event(), Event()
    now = [0.]
    worker = ContactSearchWorker(clock=lambda: now[0])
    def search():
        entered.set()
        release.wait(2.)
        raise FloatingPointError('no accepted new guess')
    try:
        assert worker.submit(search)
        assert entered.wait(1.)
        # A still-running solver never has result() called from the control loop.
        for _ in range(100):
            assert worker.poll() is None
            assert not worker.submit(search)
        release.set()
        wait([worker.future], timeout=1.)
        assert worker.poll()['retry_after'] == .25
        assert not worker.ready
        now[0] = .25
        assert worker.submit(search)
        wait([worker.future], timeout=1.)
        assert worker.poll()['retry_after'] == .5
        now[0] = .75
        assert worker.submit(lambda: 'new pair')
        wait([worker.future], timeout=1.)
        assert worker.poll()['result'] == 'new pair'
        assert worker.failures == 0
    finally:
        release.set()
        worker.close()
