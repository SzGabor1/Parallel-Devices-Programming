import cv2
import numpy as np
import pyopencl as cl

# OpenCL kernel code for matrix-vector multiplication and norm calculation
kernel_code = """
__kernel void matrix_vector_norm(__global const float *matrix,
                                 __global const float *vector,
                                 __global float *result,
                                 const int rows,
                                 const int cols) {
    int gid = get_global_id(0);
    float sum = 0.0f;
    
    for (int i = 0; i < cols; ++i) {
        int index = gid * cols + i;
        sum += matrix[index] * vector[i];
    }
    
    result[gid] = sqrt(sum);
}
"""

def read_image_as_matrix(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.astype(np.float32)

def matrix_vector_norm(matrix, vector):
    # Initialize OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Compile the kernel code
    program = cl.Program(context, kernel_code).build()
    
    # Create memory buffers for matrix, vector, and result
    matrix_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    vector_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix.shape[0] * 4)
    
    # Execute the kernel
    program.matrix_vector_norm(queue, matrix.shape, None, matrix_buffer, vector_buffer, result_buffer, np.int32(matrix.shape[0]), np.int32(matrix.shape[1]))
    
    # Retrieve the result
    result = np.empty(matrix.shape[0], dtype=np.float32)
    cl.enqueue_copy(queue, result, result_buffer).wait()
    
    return result

# Read the images and convert them into matrices
matrix1 = read_image_as_matrix("beadando/4kimages/xd.png")
matrix2 = read_image_as_matrix("beadando/4kimages/xd.png")

# Create a vector (you may need to adjust the vector based on the size of the images)
vector = np.ones(matrix1.shape[1], dtype=np.float32)

# Calculate the norms
norm1 = np.linalg.norm(matrix1.dot(vector))
norm2 = np.linalg.norm(matrix2.dot(vector))

# Calculate the percentage similarity between the norms
similarity_percentage = (min(norm1, norm2) / max(norm1, norm2)) * 100

print("Percentage similarity between the norms:", similarity_percentage, "%")
