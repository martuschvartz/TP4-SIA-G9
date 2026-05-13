from math import exp

import numpy as np

from kohonen.cords import Cords


class Neuron:
    def __init__(self, n_inputs: int, cords: Cords, weights: np.ndarray = None):
        self.n_inputs = n_inputs
        self.weights = weights if weights is not None else np.random.rand(n_inputs)
        self.cords = cords


    def update_weights(self, input_vector: np.ndarray, neighbourhood_function_value: float, learning_rate: float):
        new_weights = self.weights + neighbourhood_function_value * learning_rate * (input_vector - self.weights)
        self.weights = new_weights





class KohonenMap:
    def __init__(self, n_inputs: int, grid_size: int, init_learning_rate: float = 0.5,
                 init_radius: float = 0, dataset: np.ndarray = None, lr_decay: int = 10000):
        self.inputs = np.zeros(n_inputs, dtype=float)
        self.grid = []
        self.learning_rate = init_learning_rate
        self.radius = init_radius if init_radius != 0 else grid_size
        self.lr_decay = lr_decay

        for i in range(grid_size):
            self.grid.append([])
            for j in range(grid_size):

                # Initialize neuron weights with random dataset sample
                if dataset is not None:
                    rand_idx = np.random.randint(0, dataset.shape[0])
                    w = dataset[rand_idx].copy()
                    self.grid[i].append(Neuron(n_inputs, weights=w, cords=Cords(i, j)))
                else:
                    self.grid[i].append(Neuron(n_inputs, cords=Cords(i, j)))

    def get_neuron_at(self, cords: Cords) -> Neuron:
        return self.grid[cords.x][cords.y]

    def find_winner_cords(self, input_vector: np.ndarray) -> Cords:
        winner_cords = Cords(0,0)
        min_dist = float('inf')

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                neuron = self.grid[i][j]
                # Norm is magnitude of (input - weights) vector, so it gives us distance between input and neuron weights
                dist = np.linalg.norm(input_vector - neuron.weights)
                if dist < min_dist:
                    min_dist = dist
                    winner_cords.set_cords(i,j)

        return winner_cords

    def update_weights(self, input_vector: np.ndarray, winner_cords: Cords):
        winner_neuron = self.get_neuron_at(winner_cords)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                neuron = self.grid[i][j]
                neighbourhood_function_value = self.get_distance_to_neuron(neuron.cords, winner_neuron.cords)
                neuron.update_weights(input_vector, neighbourhood_function_value, self.learning_rate)


    def update_learning_rate(self, epoch):
        self.learning_rate = self.learning_rate * exp(-epoch / self.lr_decay)


    # In a totally arbitrary , reduce radius by 5%
    def update_radius(self):
        self.radius = max(self.radius * 0.95, 1)

    # calculate distance using eucledian distance.
    # this is the neighbourhood function
    # todo CHECK how to implement using hexagonal grid
    def get_distance_to_neuron(self, neuron_cords: Cords, winner_cords: Cords) -> float:
        if neuron_cords.x == winner_cords.x and neuron_cords.y == winner_cords.y:
            return 1
        dist = np.sqrt((neuron_cords.x - winner_cords.x) ** 2 + (neuron_cords.y - winner_cords.y) ** 2)

        #dist is always >= 1
        return 1/dist if dist < self.radius else 0