from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from proj3.objects.base import RTObject

class Sphere(RTObject):
    """A sphere object for raytracing."""
    def __init__(self, scene: 'Scene', material_index: int, 
                 center, radius: float) -> None:
        """Initialize sphere with scene, material, center and radius."""
        super().__init__(scene, material_index)
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = radius
    
    def test_intersection(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> float:
        """Find closest intersection of ray with sphere using quadratic formula."""
        # TODO: Implement ray-sphere intersection
        # 1. Calculate quadratic equation coefficients (a, b, c) using eye, direction, center, and radius
        # 2. Solve quadratic equation to find intersection points
        # 3. Return closest non-negative intersection distance or np.inf if no intersection

        oc = eye - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(direction, oc)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        #print(f"Test Intersection with a={a}, b={b}, c={c} and discriminant={discriminant}")
        if discriminant < 0:
            return np.inf
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
            elif t1 > 0:
                return t1
            elif t2 > 0:
                return t2
            else:
                return np.inf
        return np.inf
    
    def get_normal(self, point: NDArray[np.float32]) -> NDArray[np.float32]: 
    #def get_normal(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate surface normal at intersection point."""
        # TODO: Implement sphere normal calculation
        # 1. Find intersection point using test_intersection
        # 2. Calculate normal vector from sphere center to intersection point
        # 3. Normalize the vector

        normal = point - self.center
        normal /= np.linalg.norm(normal)
        return normal

        # t = self.test_intersection(eye, direction)
        # if t == np.inf:
        #     return np.zeros(3, dtype=np.float32)
        # point = eye + t * direction
        # normal = point - self.center
        # normal = normal / np.linalg.norm(normal)
        # return normal


        # return np.zeros(3, dtype=np.float32)
    

    #def get_texture_coords(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
    def get_texture_coords(self, point: NDArray[np.float32]) -> NDArray[np.float32]: 

        """Calculate spherical texture coordinates (u,v) at intersection point."""
        # TODO: Implement sphere texture coordinate calculation
        # 1. Get normal at intersection point
        # 2. Convert normal to spherical coordinates (phi, theta)
        # 3. Convert spherical coordinates to UV coordinates

        # Find intersection point

        normal = point - self.center
        normal /= np.linalg.norm(normal)
        theta = np.arccos(normal[1])  # Y is up axis
        phi = np.arctan2(normal[2], normal[0])
        if phi < 0:
            phi += 2 * np.pi
        u = phi / (2 * np.pi)
        v = theta / np.pi
        #print(f"Texture Coords: {u}, {v}")
        return np.array([u, v], dtype=np.float32)

        # t = self.test_intersection(eye, direction)
        # if t == np.inf:
        #     print(f"returning inf for t={t}")
        #     return np.array([0.0, 0.0], dtype=np.float32)
        # point = eye + t * direction

        # # Compute normal at the intersection point
        # normal = point - self.center
        # normal /= np.linalg.norm(normal)

        # # Compute spherical coordinates
        # # theta: angle from Y-axis
        # theta = np.arccos(normal[1])  # Y is up axis
        # # phi: angle from X-axis in XZ-plane
        # phi = np.arctan2(normal[2], normal[0])
        # if phi < 0:
        #     phi += 2 * np.pi

        # # Map spherical coordinates to texture coordinates (u, v)
        # u = phi / (2 * np.pi)
        # v = theta / np.pi
        # print(f"Texture Coords: {u}, {v}")
        # return np.array([u, v], dtype=np.float32)