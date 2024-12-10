import json
import re

def parse_ray_file(content):
    # Extract author from comment if present
    author = None
    author_match = re.search(r'#\s*author:\s*(\w+)', content)
    if author_match:
        author = author_match.group(1)
    
    # Initialize the JSON structure
    json_data = {
        "author": author,
        "background": {},
        "camera": {},
        "lights": [],
        "materials": [],
        "objects": {
            "spheres": [],
            "triangles": []
        }
    }
    
    # Remove comments
    content = re.sub(r'#.*?\n', '\n', content)
    
    # Parse Background
    bg_match = re.search(r'Background\s*{([^}]+)}', content)
    if bg_match:
        bg_content = bg_match.group(1)
        color_match = re.search(r'color\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', bg_content)
        if color_match:
            json_data["background"]["color"] = [float(x) for x in color_match.groups()]
        ambient_match = re.search(r'ambientLight\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', bg_content)
        if ambient_match:
            json_data["background"]["ambientLight"] = [float(x) for x in ambient_match.groups()]
    
    # Parse Camera
    cam_match = re.search(r'Camera\s*{([^}]+)}', content)
    if cam_match:
        cam_content = cam_match.group(1)
        for prop in ['eye', 'lookAt', 'up']:
            prop_match = re.search(fr'{prop}\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', cam_content)
            if prop_match:
                json_data["camera"][prop] = [float(x) for x in prop_match.groups()]
        fovy_match = re.search(r'fovy\s+([\d.]+)', cam_content)
        if fovy_match:
            json_data["camera"]["fovy"] = float(fovy_match.group(1))
    
    # Parse Lights
    lights_match = re.search(r'Lights\s*{([^}]+)}', content)
    if lights_match:
        lights_content = lights_match.group(1)
        light_pattern = r'Light\s*{([^}]+)}'
        for light_match in re.finditer(light_pattern, lights_content):
            light_data = {}
            light_content = light_match.group(1)
            
            pos_match = re.search(r'position\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', light_content)
            if pos_match:
                light_data["position"] = [float(x) for x in pos_match.groups()]
                
            color_match = re.search(r'color\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', light_content)
            if color_match:
                light_data["color"] = [float(x) for x in color_match.groups()]
                
            json_data["lights"].append(light_data)
    
    # Parse Materials
    materials_match = re.search(r'Materials\s*{([^}]+)}', content)
    if materials_match:
        materials_content = materials_match.group(1)
        material_pattern = r'Material\s*{([^}]+)}'
        for material_match in re.finditer(material_pattern, materials_content):
            material_data = {}
            material_content = material_match.group(1)
            
            # Parse texture filename
            texture_match = re.search(r'textureFilename\s+(\w+)', material_content)
            if texture_match:
                filename = texture_match.group(1)
                material_data["textureFilename"] = None if filename == "NULL" else filename
            
            # Parse colors and properties
            for prop in ['diffuseColor', 'specularColor', 'reflectiveColor', 'transparentColor']:
                color_match = re.search(fr'{prop}\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', material_content)
                if color_match:
                    material_data[prop] = [float(x) for x in color_match.groups()]
            
            # Parse scalar properties
            for prop in ['shininess', 'indexOfRefraction']:
                value_match = re.search(fr'{prop}\s+([\d.-]+)', material_content)
                if value_match:
                    material_data[prop] = float(value_match.group(1))
            
            json_data["materials"].append(material_data)
    
    # Parse Group objects
    group_match = re.search(r'Group\s*{([^}]+)}', content)
    if group_match:
        group_content = group_match.group(1)
        
        # Parse Spheres
        sphere_pattern = r'Sphere\s*{([^}]+)}'
        for sphere_match in re.finditer(sphere_pattern, group_content):
            sphere_data = {}
            sphere_content = sphere_match.group(1)
            
            material_match = re.search(r'materialIndex\s+(\d+)', sphere_content)
            if material_match:
                sphere_data["materialIndex"] = int(material_match.group(1))
                
            center_match = re.search(r'center\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', sphere_content)
            if center_match:
                sphere_data["center"] = [float(x) for x in center_match.groups()]
                
            radius_match = re.search(r'radius\s+([\d.-]+)', sphere_content)
            if radius_match:
                sphere_data["radius"] = float(radius_match.group(1))
                
            json_data["objects"]["spheres"].append(sphere_data)
        
        # Parse Triangles
        triangle_pattern = r'Triangle\s*{([^}]+)}'
        for triangle_match in re.finditer(triangle_pattern, group_content):
            triangle_data = {}
            triangle_content = triangle_match.group(1)
            
            material_match = re.search(r'materialIndex\s+(\d+)', triangle_content)
            if material_match:
                triangle_data["materialIndex"] = int(material_match.group(1))
            
            # Parse vertices
            vertices = []
            for i in range(3):
                vertex_match = re.search(fr'vertex{i}\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', triangle_content)
                if vertex_match:
                    vertices.append([float(x) for x in vertex_match.groups()])
            triangle_data["vertices"] = vertices
            
            # Parse texture coordinates
            tex_coords = []
            for i in range(3):
                tex_match = re.search(fr'tex_xy_{i}\s+([\d.-]+)\s+([\d.-]+)', triangle_content)
                if tex_match:
                    tex_coords.append([float(x) for x in tex_match.groups()])
            triangle_data["textureCoords"] = tex_coords
            
            json_data["objects"]["triangles"].append(triangle_data)
    
    return json_data

def convert_ray_to_json(ray_content):
    """Convert .ray file content to JSON string with pretty formatting."""
    json_data = parse_ray_file(ray_content)
    return json.dumps(json_data, indent=4)


import os
from pathlib import Path
from typing import List
import json

def find_ray_files(directory: str) -> List[Path]:
    """Find all .ray files in the given directory and subdirectories."""
    return list(Path(directory).rglob("*.ray"))

def convert_ray_file(ray_path: Path) -> None:
    """Convert a single .ray file to JSON and save it."""
    # Read the .ray file
    try:
        with open(ray_path, 'r') as f:
            ray_content = f.read()
    except Exception as e:
        print(f"Error reading {ray_path}: {e}")
        return

    # Convert to JSON
    try:
        json_data = parse_ray_file(ray_content)
        
        # Create output path with .json extension
        json_path = ray_path.with_suffix('.json')
        
        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Successfully converted {ray_path} to {json_path}")
    except Exception as e:
        print(f"Error converting {ray_path}: {e}")

def batch_convert_ray_files(directory: str) -> None:
    """Convert all .ray files in the given directory and its subdirectories."""
    ray_files = find_ray_files(directory)
    
    if not ray_files:
        print(f"No .ray files found in {directory}")
        return
    
    print(f"Found {len(ray_files)} .ray files")
    
    for ray_file in ray_files:
        convert_ray_file(ray_file)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert .ray files to JSON format')
    parser.add_argument('directory', help='Directory containing .ray files')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    print(f"Processing .ray files in {args.directory}")
    batch_convert_ray_files(args.directory)

if __name__ == "__main__":
    main()