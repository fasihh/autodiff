from typing import List
from node import Node


class Optimizer:
    def __init__(self, params: List[Node], lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: List[Node], lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            p.value -= self.lr * p.grad
