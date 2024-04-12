__kernel void matrix_vector_norm(__global double *matrix,
                                 __global double *vector,
                                 __global double *result, const int matrix_rows,
                                 const int matrix_columns) {
  int row = get_global_id(0);

  if (row < matrix_rows) {
    double sum = 0.0;
    for (int i = 0; i < matrix_columns; i++) {
      sum += matrix[row * matrix_columns + i];
    }
    result[row] = sqrt(sum);
  }
}