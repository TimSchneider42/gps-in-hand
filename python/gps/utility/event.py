class Event(object):
    def __init__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)

    def remove(self, handler):
        self.handlers.remove(handler)

    def fire(self, sender, *args, **kwargs):
        for handler in self.handlers:
            handler(sender, *args, **kwargs)

    __iadd__ = add
    __isub__ = remove

    def __call__(self, sender, *args, **kwargs):
        self.fire(sender, *args, **kwargs)