import numpy as np

class Light:
    """Represents a light source in the scene"""
    def __init__(self, position, color):
        self.position = np.asarray(position, dtype=np.float32)
        self.color = np.asarray(color, dtype=np.float32)
