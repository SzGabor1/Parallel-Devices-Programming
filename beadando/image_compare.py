import pyopencl as cl
import numpy as np
from PIL import Image
import os
import time
# Kernel code for image comparison
kernel_code = """
__kernel void compare_images(__global const uchar* img1,
                             __global const uchar* img2,
                             __global int* result,
                             const int width,
                             const int height)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int idx = j * width + i;

    if (i < width && j < height) {
        if (img1[idx * 4] != img2[idx * 4] ||
            img1[idx * 4 + 1] != img2[idx * 4 + 1] ||
            img1[idx * 4 + 2] != img2[idx * 4 + 2] ||
            img1[idx * 4 + 3] != img2[idx * 4 + 3]) {
            result[idx] = 0;
        }
        else {
            result[idx] = 1;
        }
    }
}
"""

# Function to load images and compare
# Function to load images and compare
def compare_images(image1_path, image2_path):
    # Load images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Convert images to RGBA format
    image1_rgba = image1.convert("RGBA")
    image2_rgba = image2.convert("RGBA")
    
    # Get image data as numpy arrays
    img1_data = np.array(image1_rgba)
    img2_data = np.array(image2_rgba)
    
    # Get image dimensions
    width, height = image1_rgba.size

    
    # Initialize OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Compile kernel code
    program = cl.Program(context, kernel_code).build()
    
    # Create image buffers
    img1_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img1_data)
    img2_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img2_data)
    
    # Allocate memory for result
    result = np.zeros(width * height, dtype=np.int32)
    result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)
    
    # Execute kernel
    event = program.compare_images(queue, (width, height), None, img1_buf, img2_buf, result_buf, np.int32(width), np.int32(height))
    
    # Read back the result
    cl.enqueue_copy(queue, result, result_buf).wait()
    
    
    is_same = False
    if np.all(result):
        print(f"{image1_path} and {image2_path} are similar.")
        is_same = True
    else:
        print(f"{image1_path} and {image2_path} are different.")
        pass

    # Get profiling information
    queued_time = event.get_profiling_info(cl.profiling_info.QUEUED)
    end_time = event.get_profiling_info(cl.profiling_info.END)
    runtime = end_time - queued_time
    
    return runtime, is_same



# Function to find similar images in a folder
def find_similar_images_gpu(image_folder, s):
    start_time = time.time()
    images = os.listdir(image_folder)
    runtime = 0
    count_same_images = 0
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            #print(f"Comparing {images[i]} and {images[j]}")
            rt, is_same = compare_images(os.path.join(image_folder, images[i]), os.path.join(image_folder, images[j]))
            runtime += rt
            if is_same:
                count_same_images += 1
            if count_same_images == s:
                    end_time = time.time()
                    #print("Total runtime:", end_time - start_time, "s")
                    #print("Total runtime on gpu:", runtime, "ns")
                    #return (end_time - start_time)
                    return(runtime)
            


        



# # Path to the folder containing images
# image_folder_path = "beadando/4kimages/"
# find_similar_images_gpu(image_folder_path)
