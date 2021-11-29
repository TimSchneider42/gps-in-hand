import time


class Timer:
    """
    Implements a simple timer based on time.time()
    """
    def __init__(self):
        self._round_start = time.time()

    def reset(self):
        """
        Resets the timer.
        :return:
        """
        self._round_start = time.time()

    def round(self):
        """
        Resets the timer and returns the current round time.
        :return:
        """
        t = time.time()
        dt = t - self._round_start
        self._round_start = t
        return dt