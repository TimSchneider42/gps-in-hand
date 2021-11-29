from threading import Lock
from typing import Optional

from gps.utility.abortable_worker import AbortableWorker
from gps.utility.event import Event
from gps.agent import Agent
from gps.agent.noise_generator import NoiseGenerator, ZeroNoiseGenerator
from gps.controller import Controller
from gps.sample import SampleList


class Sampler(AbortableWorker[SampleList]):
    def __init__(self, agent: Agent, condition: int, control_noise_generator: Optional[NoiseGenerator] = None):
        AbortableWorker.__init__(self)
        self.__agent = agent
        self.__condition = condition
        self.__control_noise_generator = ZeroNoiseGenerator() if control_noise_generator is None else \
            control_noise_generator
        self.__sample_started_event = Event()
        self.__sample_completed_event = Event()
        self.__sampling_started_event = Event()
        self.__sampling_completed_event = Event()
        self.__sampling_aborted_event = Event()
        # Required to make abort thread safe
        self.__abort_lock = Lock()

    def _run(self, controller: Controller, sample_count: int) -> Optional[SampleList]:
        self.__sampling_started_event(self, controller, sample_count)
        sample_list = []
        for i in range(sample_count):
            self.__sample_started_event(self, i)
            with self.__abort_lock:
                if self.aborting:
                    self.__sampling_aborted_event(self)
                    return None
                self.__agent.setup(controller, self.__condition, self.__control_noise_generator)
            sample = self.__agent.run()
            if self.aborting:
                self.__sampling_aborted_event(self)
                return None
            self.__sample_completed_event(self, i)
            sample_list.append(sample)
        self.__sampling_completed_event(self)
        return SampleList(sample_list)

    def _on_abort(self):
        with self.__abort_lock:
            self.__agent.abort()

    @property
    def condition(self) -> int:
        return self.__condition

    @property
    def sample_started_event(self) -> Event:
        return self.__sample_started_event

    @property
    def sample_completed_event(self) -> Event:
        return self.__sample_completed_event

    @property
    def sampling_started_event(self) -> Event:
        return self.__sampling_started_event

    @property
    def sampling_completed_event(self) -> Event:
        return self.__sampling_completed_event

    @property
    def sampling_aborted_event(self) -> Event:
        return self.__sampling_aborted_event

    @property
    def control_noise_generator(self) -> NoiseGenerator:
        return self.__control_noise_generator
