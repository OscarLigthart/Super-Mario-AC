import random


class ReplayMemory:
    """
    Experience replay
    This class stores trials and shuffles them around such that the model
    will not easily get stuck in a local optimum
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)