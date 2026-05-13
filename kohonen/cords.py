from typing import Tuple


class Cords:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __int__(self, cords: Tuple[int, int]):
        self.x = cords[0]
        self.y = cords[1]

    def get_cords(self) -> Tuple[int, int]:
        return self.x, self.y


    def set_cords(self, x: int, y: int):
        self.x = x
        self.y = y