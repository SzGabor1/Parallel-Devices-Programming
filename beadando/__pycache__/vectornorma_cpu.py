import cv2
import numpy as np

def read_image_as_matrix(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.astype(np.float32)

def matrix_vector_norm(matrix, vector):
    result = np.sqrt(np.sum(matrix * vector, axis=1))
    return result

# Read the images and convert them into matrices
matrix1 = read_image_as_matrix("beadando/4kimages/xd.png")
matrix2 = read_image_as_matrix("beadando/4kimages/xd.png")

# Create a vector (you may need to adjust the vector based on the size of the images)
vector = np.ones(matrix1.shape[1], dtype=np.float32)

# Calculate the norms
norm1 = np.linalg.norm(matrix_vector_norm(matrix1, vector))
norm2 = np.linalg.norm(matrix_vector_norm(matrix2, vector))

# Calculate the percentage similarity between the norms
similarity_percentage = (min(norm1, norm2) / max(norm1, norm2)) * 100

print("Percentage similarity between the norms:", similarity_percentage, "%")
