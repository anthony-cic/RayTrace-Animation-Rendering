import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import PIL.Image
from typing import Optional
from tqdm import tqdm  # For progress bar
import os

class Renderer:
    def __init__(self, width: int = 1280, height: int = 720, save_progress: bool = True, save_progress_path='progress_imgs', \
                 save_every_N=50) -> None:
        self.width = width
        self.height = height
        self.save_progress = save_progress
        self.save_progress_path = save_progress_path
        self.save_every_N = save_every_N
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if save_progress:
            os.makedirs(save_progress_path, exist_ok=True)
        
    def render(self, scene) -> NDArray[np.uint8]:
        """Render the scene"""
        eye = scene.get_eye()
        look_at = scene.get_look_at()
        up = scene.get_up()
        fovy = scene.get_fovy()
        
        # Convert fovy to radians
        fovy_rad = np.deg2rad(fovy)
        
        # Calculate view direction and camera basis vectors
        view_dir = look_at - eye
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # TODO: Set up camera coordinate system

        
        w = -view_dir

        # Right vector
        u = np.cross(view_dir, up)
        u /= np.linalg.norm(u)

        v = np.cross(u, view_dir)


        
        aspect_ratio = self.width / self.height
        alpha = np.tan(fovy_rad / 2)

        # Progress bar for rendering
        for y in tqdm(range(self.height)):
            for x in range(self.width):
                # TODO: Calculate ray direction for pixel (x,y)

                ndc_x = (x + 0.5) / self.width
                ndc_y = (y + 0.5) / self.height
                

                # Screen space coordinates
                pixel_camera_x = (2 * ndc_x - 1) * aspect_ratio * alpha
                pixel_camera_y = (1 - 2 * ndc_y) * alpha

                # Compute ray direction in world coordinates
                direction = pixel_camera_x * u + pixel_camera_y * v + view_dir
                direction /= np.linalg.norm(direction)
                
                # For now, just make a color pattern 
                #color = np.array([x % 255 / 255.0, y % 255 / 255.0, 0.0])
                
                # TODO: Replace with actual raytracing
                # color = scene.ray_trace(eye, direction, 0)
                

                color = scene.ray_trace(eye, direction, 0)


                # Store the color
                self.image[y, x] = np.clip(color * 255, 0, 255).astype(np.uint8)
            
            # Optionally save progress every N rows
            if self.save_progress and y % self.save_every_N == 0:
                progress_img_path = os.path.join(self.save_progress_path, f"progress_{y}.png")
                self.save_image(progress_img_path)
        
        return self.image
    
    def save_image(self, filename: str) -> None:
        """Save the current image to a file"""
        PIL.Image.fromarray(self.image).save(filename)