import os
import numpy as np
from PIL import Image
import random

def create_dummy_dataset(base_dir="dummy_dataset", num_classes=10, images_per_class=20):
    os.makedirs(base_dir, exist_ok=True)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128)
    ]
    
    for c in range(num_classes):
        class_dir = os.path.join(base_dir, f"design_{c:03d}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Create a base pattern for this design (just some geometric shapes)
        # We will vary the colors to simulate "same design, different color"
        for i in range(images_per_class):
            img = Image.new('RGB', (256, 256), color=(255, 255, 255))
            pixels = img.load()
            
            # Base color for this specific image variation
            fg_color = random.choice(colors)
            bg_color = random.choice(colors)
            
            # Draw design based on class index
            # Just simple synthetic patterns (stripes, dots, etc)
            for x in range(256):
                for y in range(256):
                    if c % 3 == 0:
                        # Vertical stripes
                        if (x // 20) % 2 == 0:
                            pixels[x, y] = fg_color
                        else:
                            pixels[x, y] = bg_color
                    elif c % 3 == 1:
                        # Checkerboard
                        if ((x // 30) + (y // 30)) % 2 == 0:
                            pixels[x, y] = fg_color
                        else:
                            pixels[x, y] = bg_color
                    else:
                        # Circles
                        if ((x-128)**2 + (y-128)**2) < (50 + c*2)**2:
                            pixels[x, y] = fg_color
                        else:
                            pixels[x, y] = bg_color
                            
            # Add some random noise
            for _ in range(1000):
                rx, ry = random.randint(0, 255), random.randint(0, 255)
                pixels[rx, ry] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
            img_path = os.path.join(class_dir, f"img_{i:03d}.jpg")
            img.save(img_path)
            
    print(f"Created dummy dataset at {base_dir} with {num_classes} classes and {images_per_class} images per class.")

if __name__ == "__main__":
    create_dummy_dataset()
