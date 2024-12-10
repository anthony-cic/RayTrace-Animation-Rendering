import json
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Optional
from pathlib import Path

from proj3.light import Light
from proj3.material import Material
from proj3.objects.rt_obj_group import RTObjGroup
from proj3.objects.sphere import Sphere
from proj3.objects.triangle import Triangle

from PIL import Image



class Scene:
    def __init__(self, filename: str):
        # Initialize with defaults 
        self.eye = np.zeros(3)
        self.look_at = np.array([0, 0, -1])
        self.up = np.array([0, 1, 0])
        self.fovy = 45.0
        
        self.bg_color = np.array([0.5, 0.5, 0.5])
        self.ambient_light = np.array([0.5, 0.5, 0.5])
        
        self.materials: List[Material] = []
        self.lights: List[Light] = []
        self.object_group = RTObjGroup()
        
        self.max_recursion_depth = 3
        #self.current_refractive_index = 1.0 

        # Parse scene file
        self._parse_scene(filename)
        print(f"Number of objects in the scene: {len(self.object_group.objects)}")

        # # Animation loading 
        # self.animations = scene_data.get('animations', {})
        # self.total_frames = self.animations.get('frames', 0)

        # self.camera_keyframes = self.animations.get('cameraAnimation', [])
        # self.object_keyframes = self.animations.get('objectAnimations', [])
        # self.light_keyframes = self.animations.get('lightAnimations', [])
        # self.material_keyframes = self.animations.get('materialAnimations', [])

    
    def _parse_scene(self, filename: str) -> None:
        """Parse a scene description file in JSON format"""
        # Check file extension
        if not filename.endswith('.json'):
            raise ValueError(f"Expected .json file, got {filename}")
            
        # Load and parse JSON file
        with open(filename, 'r') as f:
            scene_data = json.load(f)
            
        # Parse background
        if 'background' in scene_data:
            bg = scene_data['background']
            self.bg_color = np.array(bg['color'])
            self.ambient_light = np.array(bg['ambientLight'])
            
        # Parse camera
        if 'camera' in scene_data:
            cam = scene_data['camera']
            self.eye = np.array(cam['eye'])
            self.look_at = np.array(cam['lookAt'])
            self.up = np.array(cam['up'])
            self.fovy = cam['fovy']
            
        # Parse materials
        if 'materials' in scene_data:
            for mat_data in scene_data['materials']:
                has_texture = mat_data['textureFilename'] is not None
                texture_data = None
                width = 0
                height = 0
                if has_texture:
                    # Load the texture image
                    texture_image = Image.open(mat_data['textureFilename'])
                    texture_image = texture_image.convert('RGB')
                    # Convert the image to a NumPy array and normalize pixel values
                    texture_data = np.array(texture_image) / 255.0
                    height, width, _ = texture_data.shape
                else:
                    texture_data = None 


                material = Material(
                    has_texture=mat_data['textureFilename'] is not None,
                    diffuse_color=np.array(mat_data['diffuseColor']),
                    specular_color=np.array(mat_data['specularColor']),
                    reflective_color=np.array(mat_data['reflectiveColor']),
                    transparent_color=np.array(mat_data['transparentColor']),
                    shininess=mat_data['shininess'],
                    refraction_index=mat_data['indexOfRefraction'],
                    texture_data=texture_data,
                    width=width,       
                    height=height         
                )
                self.materials.append(material)
                
        # Parse lights
        if 'lights' in scene_data:
            for light_data in scene_data['lights']:
                light = Light(
                    position=np.array(light_data['position']),
                    color=np.array(light_data['color'])
                )
                self.lights.append(light)
                
        # Parse objects
        if 'objects' in scene_data:
            obj_data = scene_data['objects']

            print(f"Objects data keys: {obj_data.keys()}")

            spheres_list = obj_data.get('spheres', [])
            print(f"Number of spheres: {len(spheres_list)}")
            # Parse spheres
            for sphere_data in obj_data.get('spheres', []):
                print(f"Parsing sphere: {sphere_data}")

                sphere = Sphere(
                    scene=self,
                    material_index=sphere_data['materialIndex'],
                    center=np.array(sphere_data['center']),
                    radius=sphere_data['radius']
                )
                self.object_group.add_object(sphere)
                
            triangles_list = obj_data.get('triangles', [])
            print(f"Number of triangles: {len(triangles_list)}")
            # Parse triangles
            for tri_data in obj_data.get('triangles', []):
                print(f"Parsing triangle: {tri_data}")

                triangle = Triangle(
                    scene=self,
                    material_index=tri_data['materialIndex'],
                    p0=np.array(tri_data['vertices'][0]),
                    p1=np.array(tri_data['vertices'][1]),
                    p2=np.array(tri_data['vertices'][2]),
                    tex_coords=np.array(tri_data['textureCoords'])
                )
                self.object_group.add_object(triangle)
        else:
            print("No 'objects' key found in scene data.")
                
        if 'animations' in scene_data:
            self.animations = scene_data['animations']
            self.total_frames = self.animations.get('frames', 0)
            self.object_animations = self.animations.get('objectAnimations', [])


            self.camera_keyframes = self.animations.get('cameraAnimation', [])
            self.object_keyframes = self.animations.get('objectAnimations', [])
            self.light_keyframes = self.animations.get('lightAnimations', [])
            self.material_keyframes = self.animations.get('materialAnimations', [])
            print(f"Parsing scene with Animations {self.animations}")
        else:
            self.animations = {}
            self.total_frames = 0
            self.object_animations = []

            self.camera_keyframes = 0
            self.object_keyframes = 0
            self.light_keyframes = 0
            self.material_keyframes = 0
            print("No Animation found for this scene")

    def update_scene_for_frame(self, frame: int):
        # Example: Move sphere upward over 30 frames
        # Assuming we have 1 sphere at index 0

        for anim in self.object_animations:
            target = anim['target']  # "sphere" or "triangle"
            index = anim['index']
            keyframes = anim['keyframes']

            # Interpolate properties
            # For simplicity, assume linear interpolation:
            property_names = ['center', 'vertices']  # Add more as needed
            for prop_name in property_names:
                value = self._get_interpolated_value(frame, keyframes, prop_name)
                if value is not None:
                    # Apply this value to the object in the scene
                    if target == 'sphere':
                        spheres = [obj for obj in self.object_group.objects if isinstance(obj, Sphere)]
                        if 0 <= index < len(spheres):
                            sphere = spheres[index]
                            if prop_name == 'center':
                                sphere.center = np.array(value)
                    elif target == 'triangle':
                        triangles = [obj for obj in self.object_group.objects if isinstance(obj, Triangle)]
                        if 0 <= index < len(triangles):
                            tri = triangles[index]
                            if prop_name == 'vertices':
                                # Update triangle vertices
                                tri.p0 = np.array(value[0])
                                tri.p1 = np.array(value[1])
                                tri.p2 = np.array(value[2])
                                # Recompute edges and normals if needed
                                tri.edge1 = tri.p1 - tri.p0
                                tri.edge2 = tri.p2 - tri.p0
                                tri.normal = np.cross(tri.edge1, tri.edge2)
                                norm = np.linalg.norm(tri.normal)
                                if norm > 0:
                                    tri.normal /= norm

    def _get_interpolated_value(self, frame: int, keyframes: list, prop_name: str):
        # Find the keyframes before and after 'frame'
        sorted_kf = sorted(keyframes, key=lambda k: k['frame'])
        if frame <= sorted_kf[0]['frame']:
            return sorted_kf[0].get(prop_name)
        if frame >= sorted_kf[-1]['frame']:
            return sorted_kf[-1].get(prop_name)

        # Otherwise, interpolate
        for i in range(len(sorted_kf)-1):
            f0 = sorted_kf[i]['frame']
            f1 = sorted_kf[i+1]['frame']
            if f0 <= frame <= f1:
                v0 = sorted_kf[i].get(prop_name)
                v1 = sorted_kf[i+1].get(prop_name)
                if v0 is not None and v1 is not None:
                    t = (frame - f0) / (f1 - f0)
                    v0 = np.array(v0)
                    v1 = np.array(v1)
                    return (1 - t)*v0 + t*v1
        return None

    def refract(self, incident, normal, n1, n2):
        """Calculate the refraction direction using Snell's Law."""
        n_ratio = n1 / n2
        cos_i = -np.dot(normal, incident)
        sin_t2 = n_ratio ** 2 * (1 - cos_i ** 2)
        #print(f"Refraction calculations: n_ratio={n_ratio}, cos_i={cos_i}, sin_t2={sin_t2}")

        if sin_t2 > 1.0:
            # Total internal reflection
            return None
        cos_t = np.sqrt(1 - sin_t2)
        refract_dir = n_ratio * incident + (n_ratio * cos_i - cos_t) * normal
        refract_dir /= np.linalg.norm(refract_dir)
        return refract_dir

    def ray_trace(self, eye: NDArray[np.float32], direction: NDArray[np.float32], 
                 recurse_depth: int, current_refractive_index=1.0) -> NDArray[np.float32]:
        """Trace a ray through the scene and return the color"""
        # TODO: Implement ray tracing

        # TODO: update code structure for readability 

        if recurse_depth > self.max_recursion_depth:
            return self.bg_color

        
        EPSILON = 1e-4 #if recurse_depth > 0 else 0.0


        t_min = np.inf
        hit_object = None
        for obj in self.object_group.objects:
            t = obj.test_intersection(eye, direction)
            if t < t_min:
                t_min = t
                hit_object = obj

        if hit_object is None:
            return self.bg_color

        point = eye + t_min * direction
        normal = hit_object.get_normal(point)
        material = self.materials[hit_object.material_index]

        # Determine if the ray is inside the object
        inside = False
        if np.dot(direction, normal) > 0:
            normal = -normal
            inside = True

        # Calculate local shading
        view_dir = -direction
        local_color = self.shade(point, normal, view_dir, material, hit_object, eye)

        # Initialize total color components
        reflection_color = np.zeros(3)
        refraction_color = np.zeros(3)

        # Reflection
        if np.any(material.reflective_color > 0):
            reflect_dir = direction - 2 * np.dot(direction, normal) * normal
            reflect_dir /= np.linalg.norm(reflect_dir)
            reflection_color = self.ray_trace(point + EPSILON * reflect_dir, reflect_dir, recurse_depth + 1, current_refractive_index)
            #reflection_color *= material.reflective_color

        # if np.all(reflection_color == 0) and np.any(material.reflective_color > 0):
        #     print(f"[Warning] Reflection color is zero at depth {recurse_depth} when it should not be.")


        # Refraction
        if np.any(material.transparent_color > 0):
            n1 = current_refractive_index
            n2 = material.refraction_index if not inside else 1.0  # From material to air

            refract_dir = self.refract(direction, normal, n1, n2)
            if refract_dir is not None:
                refracted_color = self.ray_trace(point + EPSILON * refract_dir, refract_dir, recurse_depth + 1, n2)
                refraction_color = refracted_color * material.transparent_color
            else:
                # Total internal reflection
                if np.any(material.reflective_color > 0):
                    reflection_color = self.ray_trace(point + EPSILON * reflect_dir, reflect_dir, recurse_depth + 1, current_refractive_index)
                    #reflection_color *= material.reflective_color
            
        # if np.all(refraction_color == 0) and np.any(material.transparent_color > 0):
        #     print(f"[Warning] Refraction color is zero at depth {recurse_depth} when it should not be.")
        # Debug: Print color components before combining
        # print(f"Local color at depth {recurse_depth}: {local_color}")
        # print(f"Reflection color at depth {recurse_depth}: {reflection_color}")
        # print(f"Refraction color at depth {recurse_depth}: {refraction_color}")


        # Compute per-channel weights without multiplicative add ATTEMPT 2 
        total_weight = material.reflective_color + material.transparent_color
        local_weight = 1.0 - total_weight
        local_weight = np.clip(local_weight, 0, 1)
        reflect_weight = material.reflective_color
        refract_weight = material.transparent_color
        # Normalize weights if necessary
        sum_weights = local_weight + reflect_weight + refract_weight
        # Avoid division by zero
        sum_weights = np.where(sum_weights == 0, 1, sum_weights)
        local_weight /= sum_weights
        reflect_weight /= sum_weights
        refract_weight /= sum_weights
        # Compute the total color
        color = local_weight * local_color + reflect_weight * reflection_color + refract_weight * refraction_color





        # PREV WORKING SOL 
        # Compute the total color
        # local_weight = 1.0 - material.reflective_color - material.transparent_color
        # local_weight = np.clip(local_weight, 0, 1)
        # color = local_weight * local_color + reflection_color + refraction_color

        # if (color <= 0):
        #     print(f"Color neg: {color} with object {hit_object}")

        # Local shading attempt for reflec&refrac, trying to prevent neg color 
        # Calculate total weight

        # Compute the total weight new 
        # total_weight = np.sum(material.reflective_color) + np.sum(material.transparent_color)
        # local_weight = max(1.0 - total_weight, 0.0)

        # # Normalize weights
        # sum_weights = local_weight + total_weight
        # local_weight /= sum_weights
        # reflection_weight = np.sum(material.reflective_color) / sum_weights
        # refraction_weight = np.sum(material.transparent_color) / sum_weights


        # if not np.isclose(sum_weights, 1.0):
        #     print(f"[Warning] Weights do not sum to 1 at depth {recurse_depth}: sum={sum_weights}")


        #print(f"Weights at depth {recurse_depth}: local={local_weight}, reflection={reflection_weight}, refraction={refraction_weight}")

        # Compute the total color
        #color = local_weight * local_color + reflection_weight * reflection_color + refraction_weight * refraction_color
            

        # Clamp color to [0,1]
        color = np.clip(color, 0, 1)
        if np.any(color < 0) or np.any(color > 1):
            print(f"[Warning] Color out of bounds at depth {recurse_depth}: {color}")

        return color


        #print(f"t_min: {t_min}, hit_object: {hit_object}")


        #return self.bg_color

    # Accessor methods matching C++ starter code
    def get_eye(self) -> NDArray[np.float32]:
        return self.eye
        
    def get_look_at(self) -> NDArray[np.float32]:
        return self.look_at
        
    def get_up(self) -> NDArray[np.float32]:
        return self.up
        
    def get_fovy(self) -> float:
        return self.fovy

    
       
    def shade(self, point, normal, direction, material, obj, eye):
        """Compute color at point using Phong shading model"""
        view_dir = -direction  # Direction from point to eye
        color = np.zeros(3)

        # Compute ambient component
        #ambient = self.ambient_light * material.diffuse_color
       
        #print(material)

        # Compute texture color if material has a texture
        #print(f"has text: {material.has_texture}, text data: {material.texture_data} ")
        if material.has_texture and material.texture_data is not None:
            #print("material has text and data")

            tex_coords = obj.get_texture_coords(point)

            u, v = tex_coords
            #print(f"u: {u}, v: {v}")
            x = int(u * material.width)
            y = int((1 - v) * material.height)  
            # Ensure x and y are within bounds
            x = np.clip(x, 0, material.width - 1)
            y = np.clip(y, 0, material.height - 1)
            # Access the texture data at the calculated indices
            
            texture_color = material.texture_data[y, x]
            diffuse_color = texture_color
           
        else:
            diffuse_color = material.diffuse_color
        
        ambient = self.ambient_light * diffuse_color

        color += ambient

        for light in self.lights:
            # Compute light direction
            light_dir = light.position - point
            light_distance = np.linalg.norm(light_dir)
            light_dir = light_dir / light_distance
            # Diffuse component
            diff = np.maximum(np.dot(normal, light_dir), 0.0)
            diffuse = diff * diffuse_color * light.color
            # Specular component
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            spec = np.power(np.maximum(np.dot(view_dir, reflect_dir), 0.0), material.shininess)
            specular = spec * material.specular_color * light.color
            color += diffuse + specular

        # Clamp color to [0,1]
        color = np.clip(color, 0, 1)
        #print(f"final color by shade: {color}")
        return color

    