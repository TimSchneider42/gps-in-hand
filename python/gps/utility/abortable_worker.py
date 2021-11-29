# TODO: documentation
from threading import Lock
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


class AbortableWorker(Generic[T]):
    def __init__(self):
        self.__abort_lock = Lock()
        self.__aborting = False
        self.__args = None
        self.__kwargs = None
        self.__running = False
        self.__aborted = False
        # True if the worker has been setup and the setup was cleared by an an abort call before it could be run
        self.__abort_cleared_setup = False

    def _run(self, *args, **kwargs) -> T:
        pass

    def _on_abort(self):
        pass

    def setup(self, *args, **kwargs):
        with self.__abort_lock:
            if self.__running:
                raise RuntimeError("This worker is already running in another thread.")
            self.__aborting = False
            self.__aborted = False
            self.__abort_cleared_setup = False
            self.__args = args
            self.__kwargs = kwargs

    def run(self) -> Optional[T]:
        try:
            with self.__abort_lock:
                if not self.__is_setup:
                    # Don't throw an exception if the setup was cleared by an abort right before this run call
                    if self.__abort_cleared_setup:
                        self.__abort_cleared_setup = False
                        return None
                    else:
                        raise RuntimeError("This worker has not been setup.")
                if self.__running:
                    raise RuntimeError("This worker is already running in another thread.")
                self.__running = True
            return self._run(*self.__args, **self.__kwargs)
        finally:
            with self.__abort_lock:
                self.__args = self.__kwargs = None
                self.__running = False
                if self.__aborting:
                    self.__aborting = False
                    self.__aborted = True

    def abort(self):
        with self.__abort_lock:
            if self.__is_setup:
                if self.__running:
                    self.__aborting = True
                    self._on_abort()
                else:
                    self.__args = self.__kwargs = None
                    self.__abort_cleared_setup = True

    @property
    def aborting(self) -> bool:
        with self.__abort_lock:
            return self.__aborting

    @property
    def aborted(self) -> bool:
        with self.__abort_lock:
            return self.__aborted

    @property
    def is_setup(self):
        with self.__abort_lock:
            return self.__is_setup

    @property
    def __is_setup(self):
        """
        Non-thread safe version of is_setup
        :return:
        """
        return self.__args is not None

    @property
    def is_running(self):
        with self.__abort_lock:
            return self.__running
