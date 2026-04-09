import copy
import numpy as np
import collections
from fygrad.node import Node
from fygrad.module import Module, Linear
from dataclasses import dataclass

np.random.seed(42)


class NeuralNet(Module):
    def __init__(self):
        super().__init__()

        self.fc1 = Linear(6, 32, label="0")
        self.fc2 = Linear(32, 16, label="1")
        self.fc3 = Linear(16, 4, label="2")

    def forward(self, x: np.ndarray):
        x = Node.tanh(self.fc1(x))
        x = Node.tanh(self.fc2(x))
        return self.fc3(x)


@dataclass
class State:
    position: tuple[int, int]
    goal: tuple[int, int]
    nearby_obstacles: list[int]

    def flatten(self):
        scale = 10

        dx = (self.goal[0] - self.position[0]) / scale
        dy = (self.goal[1] - self.position[1]) / scale

        return np.array([[dx, dy, *self.nearby_obstacles]])


class Environment:
    def __init__(self, target: tuple[int, int]):
        self.position = (0, 0)

        self.d1 = [1, 0, -1, 0]
        self.d2 = [0, 1, 0, -1]

        self.target = target
        self.obstacles = set()
        self.visited = set()
        self.visited.add(self.position)

    def add_obstacle(self, pos: tuple[int, int]):
        self.obstacles.add(pos)

    def is_obstacle(self, pos: tuple[int, int]):
        return pos in self.obstacles

    def is_goal(self, pos: tuple[int, int]):
        return pos == self.target

    def get_state(self) -> State:
        x, y = self.position

        nearby_obstacles = [
            int((x + self.d1[i], y + self.d2[i]) in self.obstacles) for i in range(4)
        ]

        return State(self.position, self.target, nearby_obstacles)

    def step(self, action: int):
        x, y = self.position
        nx, ny = x + self.d1[action], y + self.d2[action]

        gx, gy = self.target

        dold = abs(x - gx) + abs(y - gy)
        dnew = abs(nx - gx) + abs(ny - gy)

        self.position = (nx, ny)

        # record visited position
        self.visited.add(self.position)

        reward = -0.1

        if self.is_obstacle(self.position):
            reward = -10
        elif self.is_goal(self.position):
            reward = +20
        else:
            reward += dold - dnew

        return self.get_state(), reward

    def select_action(
        self, q_values, epsilon: float = 0.0, visited_penalty: float = 1.0
    ) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(4))

        penalties = np.array(
            [
                int(
                    (self.position[0] + self.d1[i], self.position[1] + self.d2[i])
                    in self.visited
                )
                for i in range(4)
            ]
        )

        qvals = np.array(q_values).flatten().copy()
        qvals -= visited_penalty * penalties

        return int(np.argmax(qvals))


class Agent:
    def __init__(self):
        self.net = NeuralNet()

    def act(self, state: State) -> Node:
        return self.net(state.flatten())


N = 2000
gamma = 0.9
lr = 0.01
visited_penalty = 1.0


def train():
    agent = Agent()
    target_agent = Agent()
    target_agent.net = copy.deepcopy(agent.net)

    for ep in range(N):
        epsilon = max(0.05, 1 - ep / N)

        env = Environment((np.random.randint(-10, 10), np.random.randint(-10, 10)))

        for i in range(20):
            pos = (
                np.random.randint(-10, 10),
                np.random.randint(-10, 10),
            )

            if pos == env.position:
                continue

            env.add_obstacle(pos)

        for i in range(100):
            state = env.get_state()
            q_values1 = agent.act(state)

            action = env.select_action(q_values1.value, epsilon, visited_penalty)

            next_state, reward = env.step(action)

            q_values2 = target_agent.act(next_state)

            if env.is_goal(next_state.position):
                target = reward
            else:
                target = reward + gamma * np.max(q_values2.value)

            target = np.clip(target, -20, 20)

            loss: Node = (q_values1[0, action] - target) ** 2

            loss.backward()

            for param in agent.net.parameters():
                param.value -= lr * param.grad
                param.zero_grad()

            if env.is_goal(env.position):
                break

        if ep % 50 == 0:
            target_agent.net = copy.deepcopy(agent.net)

    agent.net.save("test/maze_agent.json")


env = Environment((8, 2))
env.add_obstacle((1, 0))
env.add_obstacle((0, 1))
env.add_obstacle((-1, 0))

print("Obstacles:")
print(env.obstacles)

agent = Agent()
agent.net.load("test/maze_agent.json")

print("Path:")

cost = 0
for i in range(50): # max 50 steps
    if env.position == env.target:
        break

    state = env.get_state()
    q_values = agent.act(state)

    action = env.select_action(
        q_values.value, epsilon=0.0, visited_penalty=visited_penalty
    )
    env.step(action)

    print(env.position)
    cost += 1

print("Cost:", cost)


# BFS
queue = collections.deque()
queue.append((0, 0, 0))
visited = set([(0, 0)])

while queue:
    x, y, d = queue.popleft()
    if x == env.target[0] and y == env.target[1]:
        print("BFS Cost:", d)
        break

    for i in range(4):
        x, y = x + env.d1[i], y + env.d2[i]
        if (x, y) in env.obstacles or (x, y) in visited:
            continue
        queue.append((x, y, d + 1))
        visited.add((x, y))
