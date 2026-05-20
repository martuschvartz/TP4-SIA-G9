import numpy as np

from kohonen.Coordinates import Coordinates


class Neuron:
    """
    Basic Neuron class, used to make the Kohonen Grid.
    """

    def __init__(
            self,
            n_inputs: int,
            cords: Coordinates,
            weights: np.ndarray = None
    ):
        self.n_inputs = n_inputs
        self.weights = weights if weights is not None else np.random.rand(n_inputs)
        self.cords = cords


    def update_weights(
        self,
        input_vector: np.ndarray,
        h: float,
        learning_rate: float,
    ):
        """
        Wj += h(dist) * η(i) * (Xp - Wj)

        where h(dist) is a function that controls how much a neuron
        updates based on its *distance* to the winner.

        in class -> h(dist) = 1 , Wj += η(i) * (Xp - Wj)
        """
        self.weights += h * learning_rate * (input_vector - self.weights)
