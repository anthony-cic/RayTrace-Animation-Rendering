from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from proj3.objects.base import RTObject

class RTObjGroup:
    """Group of raytracing objects"""
    def __init__(self):
        self.objects: List[RTObject] = []
        self.closest_index: Optional[int] = None
    
    def add_object(self, obj: RTObject) -> None:
        """Add object to group"""
        self.objects.append(obj)
    
    def test_intersections(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> float:
        """Find closest intersection among all objects"""
        closest = np.inf
        
        for i, obj in enumerate(self.objects):
            dist = obj.test_intersection(eye, direction)
            if dist < closest:
                closest = dist
                self.closest_index = i
                
        return closest
    
    def get_closest(self) -> Optional[RTObject]:
        """Get object that was closest in last intersection test"""
        if self.closest_index is not None:
            return self.objects[self.closest_index]
        return None
