class RunningMean(object):
    """
    Implements a metric efficiently keeping track of mean of values.
    """

    def __init__(self, batch_size=100):
        self.reset()
        self.batch_size = batch_size

    def reset(self):
        self.mean = 0
        self.size = 0

    def add(self, value):
        self.size = min(self.batch_size, (self.size + 1))
        old_mean_weighted = (1 - (1 / self.size)) * self.mean
        new_value_weighted = (1 / self.size) * value
        self.mean = old_mean_weighted + new_value_weighted

    def result(self):
        return self.mean
