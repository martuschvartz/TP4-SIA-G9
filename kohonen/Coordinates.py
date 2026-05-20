from typing import Tuple


class Coordinates:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_coordinates(self) -> Tuple[int, int]:
        return self.x, self.y


    def set_coordinates(self, x: int, y: int):
        self.x = x
        self.y = y