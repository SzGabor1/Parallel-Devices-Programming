from image_compare import find_similar_images_gpu
from image_compare_cpu import find_similar_images_cpu
from vektornorma_gpu import calculate_difference
from vektornorma_cpu import calculate_difference_cpu
import os 
import numpy as np
import cv2

def main():
    image_folder_path = "beadando/4kimages/"
    
    def read_image_as_matrix(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.astype(float) 
    
    # Function to find similar images in a folder


    def load_images(image_folder):
        images = os.listdir(image_folder)
        img_arr = []
        img_paths = []
        loaded_images = set()  # Set to keep track of loaded image filenames

        for i in range(len(images)):
            if images[i] not in loaded_images:  # Check if the image has already been loaded
                img_arr.append(read_image_as_matrix(os.path.join(image_folder, images[i])))
                img_paths.append(os.path.join(image_folder, images[i]))
                loaded_images.add(images[i])  # Add the image filename to the set of loaded images

        print("LOADED IMAGES")
        return img_arr, img_paths

        
    img_arr, img_paths = load_images(image_folder_path)

    #print(img_paths)

    for i in range(len(img_arr)):
        for j in range(i + 1, len(img_arr)):
            if(img_arr[i].shape == img_arr[j].shape):
                if j > i:
                    print(f"Comparing {img_paths[i]} and {img_paths[j]}")
                    print("GPU runtime:",calculate_difference(img_arr[i], img_arr[j])/1e6)
                    print("CPU runtime:", calculate_difference_cpu(img_arr[i], img_arr[j]))

if __name__ == "__main__":
    main()
