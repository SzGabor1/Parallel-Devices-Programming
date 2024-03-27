from PIL import Image
import os
import time
import numpy as np

# Function to compare two images
def compare_images(image1_path, image2_path):
    # Load images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Convert images to RGBA format
    image1_rgba = image1.convert("RGBA")
    image2_rgba = image2.convert("RGBA")
    
    # Get image size
    width, height = image1_rgba.size
    
    # Start time
    
    is_same = True  # Assume images are the same
    
    # Get image data as numpy arrays
    img1_data = np.array(image1_rgba)
    img2_data = np.array(image2_rgba)
    
    start_time = time.time()
    # Compare images pixel-wise using numpy arrays
    if not np.array_equal(img1_data, img2_data):
        is_same = False
    
    end_time = time.time()
    return end_time - start_time, is_same  # Return the runtime and whether images are same

# Function to find similar images in a folder
def find_similar_images_cpu(image_folder, s):
    images = os.listdir(image_folder)
    total_runtime = 0
    count_same_images = 0
    
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            result = compare_images(os.path.join(image_folder, images[i]), os.path.join(image_folder, images[j]))
            
            if isinstance(result, tuple):  # Check if result is a tuple
                runtime, is_same = result
                if is_same:
                    count_same_images += 1
                total_runtime += runtime
                
                if count_same_images == s:
                    return total_runtime
