from numpy.typing import NDArray
import numpy as np

class Material:
    """Represents material properties for raytracing"""
    def __init__(
        self,
        has_texture: bool = False,
        texture_data = None,
        width: int = 0,
        height: int = 0,
        diffuse_color = None,
        specular_color = None,
        transparent_color = None,
        reflective_color = None,
        shininess: float = 1.0,
        refraction_index: float = 1.0
    ):
        # Whether material uses an image texture
        self.has_texture = has_texture
        
        # Image data if texture is present
        self.texture_data = texture_data
        self.width = width
        self.height = height
        
        # Basic material properties with defaults
        self.diffuse_color = diffuse_color if diffuse_color is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.specular_color = specular_color if specular_color is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.transparent_color = transparent_color if transparent_color is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.reflective_color = reflective_color if reflective_color is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shininess = shininess
        self.refraction_index = refraction_index