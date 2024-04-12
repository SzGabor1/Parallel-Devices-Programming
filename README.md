# Image Comparison

## Overview

The task involves performing image comparison using OpenCL on both GPU and CPU, and analyzing the execution times of each method.

## Methodology

The task is divided into two parts:

1. **Pixel-by-Pixel Comparison**:
   - Images are extracted from a folder and converted into matrices.
   - Matrices are compared pixel by pixel.
   - Accurate representation of similarities.
   - Time-consuming for large images.

![Runtime Graph](https://github.com/SzGabor1/Parallel-Devices-Programming/blob/main/beadando/runtime.png)


2. **Matrix Vector Norms**:
   - Estimates similarity using matrix vector norms.
   - Returns the result as a percentage.
   - Allows the user to determine the acceptable level of similarity.
   - Considers images identical if identical values appear in a row but in a different order.
   - Potential solutions include processing by columns or examining images using the pixel-by-pixel comparison approach after achieving 100% similarity.

## Pros and Cons

- **Pixel-by-Pixel Comparison**:
  - *Pros*: Accurate representation of similarities.
  - *Cons*: Time-consuming for large images.

- **Matrix Vector Norms**:
  - *Pros*: Provides similarity estimate as a percentage.
  - *Cons*: May consider images identical even if values are in different order within a row.

## Future Improvements

- Enhance efficiency of pixel-by-pixel comparison method, possibly through parallel processing techniques.
- Refine matrix vector norms method to handle cases where values appear in different orders within a row.

## Usage

1. Clone the repository.
2. Run the desired comparison method script.
3. Edit the path of the image folder.
4. Analyze results, including execution times and similarity estimates.
