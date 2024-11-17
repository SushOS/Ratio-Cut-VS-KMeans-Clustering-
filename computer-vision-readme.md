# Image Segmentation Using K-means and Ratio-Cut Clustering

This repository contains the implementation of image segmentation techniques using K-means and Ratio-Cut (Spectral) clustering methods. The project is part of the Computer Vision course (CSL7360).

## Introduction

This project implements and compares two clustering techniques for image segmentation:
- Ratio-Cut (Spectral Clustering)
- K-means Clustering

The goal is to partition images into meaningful regions by grouping pixels that share similar characteristics.

## Implementation Details

### K-means Clustering
- Implements traditional K-means clustering for image segmentation
- Segments images using both k=3 and k=6 clusters
- Works directly with RGB color values

### Ratio-Cut (Spectral) Clustering
1. **Pre-processing**:
   - Converts RGB images to grayscale
   - Resizes images to 64x64 pixels

2. **Algorithm Steps**:
   - Computes affinity matrix using Gaussian kernel
   - Calculates degree matrix
   - Constructs Laplacian matrix
   - Performs eigenvalue decomposition
   - Applies K-means clustering on eigenvectors

## Features

- Image preprocessing and resizing functionality
- Support for multiple clustering parameters (k=3 and k=6)
- Visualization of segmentation results
- Comparative analysis between different methods

## Requirements

- OpenCV (cv2)
- NumPy
- Matplotlib
- Python 3.x

## Usage

1. Load your images:
```python
images = load_imgs(folder_path)
```

2. Preprocess images:
```python
resized_images = resize_images(images, size=(64, 64))
```

3. Apply clustering:
```python
# For K-means
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(flattened_pixels)

# For Ratio-Cut
# Follow the spectral clustering implementation steps
```

## Results

The repository includes comparisons between:
- K-means clustering (k=3)
- K-means clustering (k=6)
- Ratio-Cut clustering (k=3)
- Ratio-Cut clustering (k=6)

### Sample Results
- Original images are segmented using both methods
- Visual comparisons demonstrate the effectiveness of each approach
- Different k values show varying levels of segmentation detail

## Author
Sushant Ravva (B21CS084)

## Course Information
Computer Vision (CSL7360)

## Note
This is an academic project focused on implementing and comparing different image segmentation techniques. The implementation is designed for educational purposes and can be extended for more complex applications.
