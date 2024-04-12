import numpy as np
import pyopencl as cl
import random
import cv2

def load_kernel_source(path):
    try:
        with open(path, 'r') as source_file:
            kernel_code = source_file.read()
        return kernel_code
    except FileNotFoundError:
        return None

# OpenCL kernel code for matrix-vector multiplication and norm calculation


def matrix_vector_norm(matrix, vector):
    # Initialize OpenCL
    kernel_code = load_kernel_source(r'beadando/kernel/vecnorm.cl')
    
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Enable profiling
    
    # Compile the kernel code
    program = cl.Program(context, kernel_code).build()
    
    # Create memory buffers for matrix, vector, and result
    matrix_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    vector_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix.size * 8) # double size is 8 bytes
    
    # Execute the kernel
    event = program.matrix_vector_norm(queue,  matrix.shape, None, matrix_buffer, vector_buffer, result_buffer, np.int32(matrix.shape[0]), np.int32(matrix.shape[1])) # capture event
    
    # Wait for the kernel to finish
    event.wait()
    
    # Retrieve the result
    result = np.empty(matrix.shape[0], dtype=float)
    cl.enqueue_copy(queue, result, result_buffer).wait()
    
    try:
        # Get profiling information
        queued_time = event.get_profiling_info(cl.profiling_info.QUEUED)
        end_time = event.get_profiling_info(cl.profiling_info.END)
        runtime = end_time - queued_time
    except cl.RuntimeError as e:
        print("Error getting profiling information:", e)
        runtime = None
    
    return result, runtime

def calculate_difference(image_matrix1, image_matrix2):
    image_vector1 = np.ones(len(image_matrix1), dtype=float)
    img1_mvn, runtime = matrix_vector_norm(image_matrix1, image_vector1)
    with open('mvn_gpu.txt', 'w') as f:
        for item in img1_mvn:
            f.write("%s\n" % item)
    image_vector2 = np.ones(len(image_matrix2), dtype=float)
    img2_mvn, runtime2 = matrix_vector_norm(image_matrix2, image_vector2)
    with open('mvn2_gpu.txt', 'w') as f:
        for item in img2_mvn:
            f.write("%s\n" % item)
    
    total_runtime = runtime + runtime2
    #print("Total runtime:", total_runtime / 1e6, "ms")
    
    # Calculate the element-wise minimum and maximum between img1_mvn and img2_mvn
    min_norm = np.minimum(img1_mvn, img2_mvn)
    max_norm = np.maximum(img1_mvn, img2_mvn)

    # print("Min norm:", img1_mvn)
    # print("Max norm:", img2_mvn)
    
    # Calculate the percentage similarity between the norms for each pair
    similarity_percentage = (min_norm / max_norm) * 100    

    # Calculate the overall percentage similarity as the average of all the similarities
    overall_similarity_percentage = np.mean(similarity_percentage)
    
    print("Overall percentage similarity between the norms:", overall_similarity_percentage, "%")
    return total_runtime

# def read_image_as_matrix():
#         image1 = cv2.imread('beadando/4kimages/1327999.jpeg', cv2.IMREAD_GRAYSCALE)
#         image2 = cv2.imread('beadando/4kimages/1327999.jpeg', cv2.IMREAD_GRAYSCALE)
#         return image1.astype(float), image2.astype(float)  
    
# matrix1, matrix2 = read_image_as_matrix()
# calculate_difference(matrix1, matrix2)
