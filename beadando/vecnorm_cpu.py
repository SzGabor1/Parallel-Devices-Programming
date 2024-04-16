import math
import cv2
import numpy as np
import time


def matrix_vector_norm(matrix):
    matrix = np.array(matrix).astype(float)
    matrix_rows = np.int32(matrix.shape[0])
    matrix_columns = np.int32(matrix.shape[1])
    result = []
    
    for row in range(matrix_rows):
        sum = 0.0
        for i in range(matrix_columns):
            sum += matrix[row, i]
        result.append(np.sqrt(sum))
    
    return result


def calculate_difference_cpu(image_matrix1, image_matrix2):
    image1_start = time.time()
    img1_mvn = matrix_vector_norm(image_matrix1)
    image1_end = time.time()
    with open('mvn_cpu.txt', 'w') as f:
        for item in img1_mvn:
            f.write("%s\n" % item)
    
    image2_start = time.time()
    img2_mvn = matrix_vector_norm(image_matrix2)
    image2_end = time.time()
    with open('mvn2_cpu.txt', 'w') as f:
        for item in img2_mvn:
            f.write("%s\n" % item)
    total_runtime = ((image1_end - image1_start) + (image2_end - image2_start))*1000
    
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
    print("CPU runtime:", total_runtime, "ms")
    return total_runtime

# def read_image_as_matrix():
#         image1 = cv2.imread('beadando/4kimages/xd.png', cv2.IMREAD_GRAYSCALE)
#         image2 = cv2.imread('beadando/4kimages/xd2.png', cv2.IMREAD_GRAYSCALE)
#         return image1.astype(float), image2.astype(float)  
    
# matrix1, matrix2 = read_image_as_matrix()
# calculate_difference_cpu(matrix1, matrix2)