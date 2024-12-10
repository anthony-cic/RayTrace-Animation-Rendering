from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class RTObject(ABC):
    """Base class for all raytraced objects"""
    def __init__(self, scene: 'Scene', material_index: int) -> None:
        self.scene = scene
        self.material_index = material_index
    
    @abstractmethod
    def test_intersection(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> float:
        """Test if ray intersects with object.
        
        Args:
            eye: Origin point of ray as (x,y,z) array
            direction: Normalized direction vector of ray as (x,y,z) array
            
        Returns:
            Distance to intersection point or np.inf if no intersection
        """
        pass
    
    @abstractmethod
    def get_normal(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get surface normal at intersection point.
        
        Args:
            eye: Origin point of ray as (x,y,z) array
            direction: Normalized direction vector of ray as (x,y,z) array
            
        Returns:
            Normalized normal vector at intersection point
        """
        pass
        
    @abstractmethod
    def get_texture_coords(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get texture coordinates at intersection point.
        
        Args:
            eye: Origin point of ray as (x,y,z) array
            direction: Normalized direction vector of ray as (x,y,z) array
            
        Returns:
            (u,v) texture coordinates in range [0,1]
        """
        pass