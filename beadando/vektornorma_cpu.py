import math
import cv2
import numpy as np


def matrix_vector_norm(matrix, vector):
    matrix = np.array(matrix).astype(np.float32)
    vector = np.array(vector).astype(np.float32)
    matrix_rows, matrix_columns = matrix.shape
    result = np.zeros(matrix_rows, dtype=np.float32)
    
    for row in range(matrix_rows):
        sum = np.float32(0.0)
        for i in range(matrix_columns):
            sum += matrix[row, i] * vector[i]
        result[row] = sum
    
    return result


def calculate_difference_cpu(image_matrix1, image_matrix2):
    image_vector1 = [1] * len(image_matrix1[0])
    img1_mvn = matrix_vector_norm(image_matrix1, image_vector1)
    with open('mvn_cpu.txt', 'a') as f:
        for item in img1_mvn:
            f.write("%s\n" % item)
    
    image_vector2 = [1] * len(image_matrix2[0])
    img2_mvn = matrix_vector_norm(image_matrix2, image_vector2)
    with open('mvn2_cpu.txt', 'a') as f:
        for item in img2_mvn:
            f.write("%s\n" % item)
    total_runtime = 0
    
    # Calculate the element-wise minimum and maximum between img1_mvn and img2_mvn
    min_norm = [min(x, y) for x, y in zip(img1_mvn, img2_mvn)]
    max_norm = [max(x, y) for x, y in zip(img1_mvn, img2_mvn)]
    
    # Calculate the percentage similarity between the norms for each pair
    similarity_percentage = [(x / y) * 100 for x, y in zip(min_norm, max_norm)]
    
    # Calculate the overall percentage similarity as the average of all the similarities
    overall_similarity_percentage = sum(similarity_percentage) / len(similarity_percentage)
    
    print("Overall percentage similarity between the norms:", overall_similarity_percentage, "%")
    return total_runtime

def read_image_as_matrix():
        image1 = cv2.imread('beadando/4kimages/xd.png', cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread('beadando/4kimages/xd2.png', cv2.IMREAD_GRAYSCALE)
        return image1.astype(np.float32), image2.astype(np.float32)  
    
matrix1, matrix2 = read_image_as_matrix()
calculate_difference_cpu(matrix1, matrix2)