import argparse
import PIL.Image
import os

from proj3.scene import Scene
from proj3.render import Renderer

def main():
    parser = argparse.ArgumentParser(description="Raytracer") 
    parser.add_argument("--ray_path", type=str, default='Media/SampleScenes/reflectionTest.json', help="Path to the scene file")
    parser.add_argument("--width", type=int, default=1280, help="Width of the output image")
    parser.add_argument("--height", type=int, default=720, help="Height of the output image")
    parser.add_argument("--save_progress", action='store_true', help="Save progress images")
    parser.add_argument("--max_depth", type=int, default=5, help="Max depth of recursion")
    args = parser.parse_args()
    
    scene = Scene(args.ray_path)
    
    scene_name = args.ray_path.split('/')[-1].split('.')[0] # You may need to change this
    renderer = Renderer(args.width, args.height, save_progress=args.save_progress, save_progress_path=f"progress_imgs_{scene_name}", save_every_N=50)

    total_frames = scene.total_frames if scene.total_frames > 0 else 30  # default to 30 frames if not specified
    print(f"total frames: {total_frames}")
    for frame in range(total_frames):
        scene.update_scene_for_frame(frame)
        final_image = renderer.render(scene)


        # Save each frame as a PNG
        os.makedirs("frames", exist_ok=True)
        frame_path = f"frames/{scene_name}_frame_{frame:03d}.png"
        PIL.Image.fromarray(final_image).save(frame_path)
        print(f"Saved frame {frame} to {frame_path}")

    #final_image = renderer.render(scene) # old final image 
    
    # Save final result
    os.makedirs("final", exist_ok=True)
    final_img_path = f"final/{scene_name}.png"
    PIL.Image.fromarray(final_image).save(final_img_path)

if __name__ == "__main__":

    main()