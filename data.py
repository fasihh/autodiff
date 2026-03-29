import numpy as np


class DataLoader:
    def __init__(
        self,
        X, y,
        batch_size=32,
        shuffle=True,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = np.arange(len(self.X))

        if self.shuffle:
            np.random.shuffle(idx)

        for i in range(0, len(idx), self.batch_size):
            batch = idx[i : i + self.batch_size]
            X = self.X[batch]
            y = self.y[batch]
            yield X, y
