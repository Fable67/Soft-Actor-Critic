class RunningMean(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_total = 0
        self.size = 0
        self.average = 0

    def add(self, value):
        self.prev_total += value
        self.size += 1
        self.average = self.prev_total / self.size

    def result(self):
        return self.average
