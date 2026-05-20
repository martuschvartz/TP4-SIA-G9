from abc import ABC, abstractmethod

import numpy as np

from kohonen.Coordinates import Coordinates
from kohonen.Neuron import Neuron


class Grid(ABC):
    """
    Class that represent a grid of Neurons, exposes TWO main methods:
    - neighbors(coords)        -> list of adjacent Neurons      -> used for U-MAP
    - distance(coords, coords) -> distance between two Neurons  -> used to determine neighbors in weight update
    """

    def __init__(self, grid_size: int, input_per_neuron: int, dataset: np.ndarray = None):
        self.grid_size = grid_size
        self.input_per_neuron = input_per_neuron
        self._grid: list[list[Neuron]] = []

        for i in range(grid_size):
            self._grid.append([])
            for j in range(grid_size):
                if dataset is not None:
                    rand_idx = np.random.randint(0, dataset.shape[0])
                    w = dataset[rand_idx].copy()
                    self._grid[i].append(Neuron(input_per_neuron, weights=w, cords=Coordinates(i, j)))
                else:
                    self._grid[i].append(Neuron(input_per_neuron, cords=Coordinates(i, j)))


    def get_neuron(self, x: int, y: int) -> Neuron:
        return self._grid[x][y]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _neurons_at(self, candidates: list[list[int]]) -> list[Neuron]:
        """Filter a list of [x, y] pairs to in-bounds cells and return their Neurons."""
        return [
            self._grid[x][y]
            for x, y in candidates
            if self._in_bounds(x, y)
        ]

    def all_neurons(self):
        for row in self._grid:
            yield from row

    @abstractmethod
    def neighbors(self, target: Coordinates) -> list[Neuron]:
        """
        Return the list of neurons adjacent to `target` (excluding target itself).
        Used for U-MAP calculation
        """

    @abstractmethod
    def distance(self, a: Coordinates, b: Coordinates) -> float:
        """
        Return the grid distance between two cells.
        Used for updating neighborhood weights
        """



class SquareGrid(Grid):
    """
    Standard rectangular grid with 4-connected neighbours.
    Distance is Euclidean: sqrt((dx)^2 + (dy)^2).
    """

    def neighbors(self, target: Coordinates) -> list[Neuron]:
        x, y = target.x, target.y
        candidates = [
                             [x, y - 1],
            [x - 1, y    ],              [x + 1, y    ],
                             [x, y + 1],
        ]
        return self._neurons_at(candidates)

    def distance(self, a: Coordinates, b: Coordinates) -> float:
        return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))



class HexGrid(Grid):
    """
    Hexagonal grid using offset coordinates (odd-columns shifted).
    Neighbors follow the standard 6-connectivity for hex grids.
    Distance is computed in axial (cube) coordinates for correctness.

    Offset → axial conversion (odd-q):
        q = x
        r = y - (x - (x & 1)) // 2
    Hex distance (axial):
        (|dq| + |dq + dr| + |dr|) / 2
    """

    def neighbors(self, target: Coordinates) -> list[Neuron]:
        x, y = target.x, target.y
        if x % 2 == 0:
            candidates = [
                [x,     y - 1],
                [x - 1, y    ], [x + 1, y    ],
                [x - 1, y + 1], [x + 1, y + 1],
                [x,     y + 1],
            ]
        else:
            candidates = [
                [x,     y - 1],
                [x - 1, y - 1], [x + 1, y - 1],
                [x - 1, y    ], [x + 1, y    ],
                [x,     y + 1],
            ]
        return self._neurons_at(candidates)

    def _to_axial(self, cords: Coordinates) -> tuple[int, int]:
        q = cords.x
        r = cords.y - (cords.x - (cords.x & 1)) // 2
        return q, r

    def distance(self, a: Coordinates, b: Coordinates) -> float:
        aq, ar = self._to_axial(a)
        bq, br = self._to_axial(b)
        dq, dr = aq - bq, ar - br
        return (abs(dq) + abs(dq + dr) + abs(dr)) / 2


def make_grid(grid_type: str, grid_size: int, input_per_neuron: int, dataset: np.ndarray = None) -> Grid:
    """
    Create the right Grid subclass from a string tag.
    Accepted values: "square", "hex"
    """
    match grid_type.lower():
        case "square":
            return SquareGrid(grid_size, input_per_neuron, dataset)
        case "hex":
            return HexGrid(grid_size, input_per_neuron, dataset)
        case _:
            raise ValueError(f"Unknown grid type '{grid_type}'.")
