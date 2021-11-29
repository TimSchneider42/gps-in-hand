from logging import Handler, INFO, LogRecord


class StatusMessageHandler(Handler):
    def __init__(self, initial_message: str = None):
        super(StatusMessageHandler, self).__init__(level=INFO)
        self._message = initial_message

    def emit(self, record: LogRecord):
        self._message = record.getMessage()

    @property
    def message(self) -> str:
        return self._message
