from math import exp

import numpy as np

from kohonen.Coordinates import Coordinates
from kohonen.Grid import Grid, make_grid


class KohonenMap:
    def __init__(
        self,
        n_inputs: int,
        grid_size: int,
        radius_decay: int, # convention is epochs / 2
        lr_decay: int,      # convention is epochs * 10
        grid_type: str = "square",
        init_learning_rate: float = 0.5,
        init_radius: float = 0,
        dataset: np.ndarray = None,
    ):
        self.init_radius = init_radius if init_radius != 0 else float(grid_size)
        self.radius = self.init_radius
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate

        self.radius_decay = radius_decay
        self.lr_decay = lr_decay

        self.grid: Grid = make_grid(grid_type, grid_size, n_inputs, dataset)


    def find_winner_cords(self, input_vector: np.ndarray) -> Coordinates:
        """
        Returns the coordinates of the best matching unit, using Euclidean distance as
        medida de similitud *d* (slide 25)
        """
        winner_cords = Coordinates(0, 0)
        min_dist = float("inf")
        for neuron in self.grid.all_neurons():
            dist = np.linalg.norm(input_vector - neuron.weights) # <----- here!
            if dist < min_dist:
                min_dist = dist
                winner_cords.set_coordinates(neuron.cords.x, neuron.cords.y)
        return winner_cords


    def update_weights(self, input_vector: np.ndarray, winner_cords: Coordinates):
        """
        Updates weights after finding BMU for each neuron in its neighborhood.

        There is no hard cutoff like:
            h(d) =  gaussian_value      if   d <= radius
                    0                   if   d > radius


        - The "cutoff" is the threshold: everyone INSIDE THE THRESHOLD receives an update
        - Threshold symbolizes neuron's influence: those with very low h (h < 1e-4) have little to no influence so they are excluded
        """
        for neuron in self.grid.all_neurons():
            h = self._neighbor_h(neuron.cords, winner_cords)
            if h > 1e-4:
                neuron.update_weights(input_vector, h, self.learning_rate)


    def _neighbor_h(self, neuron_cords: Coordinates, winner_cords: Coordinates) -> float:
        """
        Gaussian neighborhood function: h(d) = exp(-d² / (2 * radius²))
        :returns:
            - h in (0,1) for any neuron INSIDE radius, decreases smoothly with distance
            - h = 1 for BMU
        """
        dist = self.grid.distance(neuron_cords, winner_cords)
        return exp(-(dist ** 2) / (2 * self.radius ** 2))

    def update_learning_rate(self, epoch: int):
        self.learning_rate = self.init_learning_rate * exp(-epoch / self.lr_decay)


    def update_radius(self, epoch: int):
        self.radius = max(self.init_radius * exp(-epoch / self.radius_decay),1.0)
