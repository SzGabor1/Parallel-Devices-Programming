from image_compare import find_similar_images_gpu
from image_compare_cpu import find_similar_images_cpu
from vecnorm_gpu import calculate_difference
from vecnorm_cpu import calculate_difference_cpu
import os 
import numpy as np
import cv2

def main():
    image_folder_path = "beadando/images/"
    
    def read_image_as_matrix(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.astype(float) 
    



    def load_images(image_folder):
        images = os.listdir(image_folder)
        img_arr = []
        img_paths = []
        loaded_images = set() 

        for i in range(len(images)):
            if images[i] not in loaded_images:
                img_arr.append(read_image_as_matrix(os.path.join(image_folder, images[i])))
                img_paths.append(os.path.join(image_folder, images[i]))
                loaded_images.add(images[i]) 

        print("LOADED IMAGES")
        return img_arr, img_paths

        
    img_arr, img_paths = load_images(image_folder_path)

    #print(img_paths)

    # Assuming img_arr, img_paths, calculate_difference, and calculate_difference_cpu are defined elsewhere

    # Open a file in write mode
    with open("beadando/image_runtimes.txt", "w") as f:
        # Iterate over pairs of images
        for i in range(len(img_arr)):
            for j in range(i + 1, len(img_arr)):
                if img_arr[i].shape == img_arr[j].shape:
                    if j > i:
                        # Write data to the file
                        f.write(f"Comparing {img_paths[i]} and {img_paths[j]}\n")
                        f.write("GPU runtime: {}\n".format(calculate_difference(img_arr[i], img_arr[j])/1e6))
                        f.write("CPU runtime: {}\n".format(calculate_difference_cpu(img_arr[i], img_arr[j])))
                        f.write("\n")


if __name__ == "__main__":
    main()
