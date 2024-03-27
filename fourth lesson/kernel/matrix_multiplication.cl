__kernel void matrix_multiplication(__global const float *A,
                                    __global const float *B, __global float *C,
                                    const int size) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  float sum = 0.0f;
  for (int k = 0; k < size; ++k) {
    sum += A[row * size + k] * B[k * size + col];
  }

  C[row * size + col] = sum;
}
