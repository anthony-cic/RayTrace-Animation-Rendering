import numpy as np
from numpy.typing import NDArray
from proj3.objects.base import RTObject

class Triangle(RTObject):
    """A triangle object for raytracing."""
    def __init__(self, scene: 'Scene', material_index: int,
                 p0, p1, p2, tex_coords) -> None:
        """Initialize triangle with vertices and texture coordinates.
        
        Args:
            scene: Scene containing this triangle
            material_index: Index of material to use
            p0, p1, p2: Triangle vertices as (x,y,z) arrays
            tex_coords: Texture coordinates, shape (3,2) for (u,v) per vertex
        """
        super().__init__(scene, material_index)
        
        # Convert points to numpy arrays
        self.p0 = np.asarray(p0, dtype=np.float32)
        self.p1 = np.asarray(p1, dtype=np.float32)
        self.p2 = np.asarray(p2, dtype=np.float32)
        self.tex_coords = np.asarray(tex_coords, dtype=np.float32)
        
        # Pre-compute triangle edges and normal
        self.edge1 = self.p1 - self.p0
        self.edge2 = self.p2 - self.p0
        self.normal = np.cross(self.edge1, self.edge2)
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal /= norm
    
    def test_intersection(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> float:
        """Test ray intersection with triangle."""
        # TODO: Implement ray-triangle intersection
        # 1. Calculate intersection of ray with triangle plane
        # 2. Check if intersection point lies within triangle using barycentric coordinates
        # 3. Return intersection distance or np.inf if no intersection

        epsilon = 1e-6
        edge1 = self.edge1
        edge2 = self.edge2
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)
        if -epsilon < a < epsilon:
            return np.inf  # Ray is parallel to triangle
        f = 1.0 / a
        s = eye - self.p0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return np.inf
        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            return np.inf
        t = f * np.dot(edge2, q)
        if t > epsilon:
            # self.last_u = u
            # self.last_v = v
            return t
        else:
            return np.inf
    
    def get_normal(self, point: NDArray[np.float32]) -> NDArray[np.float32]:
    #def get_normal(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return pre-computed triangle normal."""
        return self.normal
        
    #def get_texture_coords(self, eye: NDArray[np.float32], direction: NDArray[np.float32]) -> NDArray[np.float32]:
    def get_texture_coords(self, point: NDArray[np.float32]) -> NDArray[np.float32]: 

        """Calculate texture coordinates using barycentric coordinates."""
        # TODO: Implement triangle texture coordinate calculation
        # 1. Find intersection point
        # 2. Calculate barycentric coordinates
        # 3. Use barycentric coordinates to interpolate texture coordinates

        # Compute vectors
        v0 = self.p1 - self.p0
        v1 = self.p2 - self.p0
        v2 = point - self.p0

        # Compute dot products
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        # Compute denominator
        denom = d00 * d11 - d01 * d01

        if denom == 0:
            # Degenerate triangle
            return np.array([0.0, 0.0], dtype=np.float32)

        # Compute barycentric coordinates
        inv_denom = 1.0 / denom
        u = (d11 * d20 - d01 * d21) * inv_denom
        v = (d00 * d21 - d01 * d20) * inv_denom
        w = 1.0 - u - v

        # Ensure u, v, w are valid
        if u < 0 or v < 0 or w < 0:
            # The point is outside the triangle
            return np.array([0.0, 0.0], dtype=np.float32)

        # Interpolate texture coordinates
        tex_coord = w * self.tex_coords[0] + u * self.tex_coords[1] + v * self.tex_coords[2]
        return tex_coord



        # # Find intersection point
        # if not hasattr(self, 'last_u') or not hasattr(self, 'last_v'):
        #     # If last_u and last_v are not set, you need to call test_intersection
        #     t = self.test_intersection(eye, direction)
        #     if t == np.inf:
        #         return np.array([0.0, 0.0], dtype=np.float32)
        # u = self.last_u
        # v = self.last_v
        # w = 1 - u - v
        # # Interpolate texture coordinates
        # tex_coord = w * self.tex_coords[0] + u * self.tex_coords[1] + v * self.tex_coords[2]
        # #print(f"Texture coords: {tex_coord}")
        # return tex_coord