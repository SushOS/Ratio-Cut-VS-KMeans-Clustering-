# Image Segmentation Using K-means and Ratio-Cut Clustering

This repository contains the implementation of image segmentation techniques using K-means and Ratio-Cut (Spectral) clustering methods. The project is part of the Computer Vision course (CSL7360).

## Introduction

This project implements and compares two clustering techniques for image segmentation:
- Ratio-Cut (Spectral Clustering)
- K-means Clustering

The goal is to partition images into meaningful regions by grouping pixels that share similar characteristics.

## Implementation Details

### K-means Clustering
```python
def kmeans_segmentation(image, k=3):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reconstruct the image with cluster centers
    segmented = kmeans.cluster_centers_[labels]
    return segmented.reshape(image.shape)
```

### Ratio-Cut (Spectral) Clustering
```python
def spectral_clustering(image, k=3):
    # Convert to grayscale
    gray = rgb2gray(image)
    
    # Compute affinity matrix
    affinity = compute_affinity_matrix(gray)
    
    # Calculate degree matrix
    degree = np.diag(np.sum(affinity, axis=1))
    
    # Construct Laplacian matrix
    laplacian = degree - affinity
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    
    # Select k eigenvectors
    features = eigvecs[:, :k]
    
    # Apply KMeans to eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return labels.reshape(image.shape[:2])
```

## Features

### Image Preprocessing
```python
def resize_images(images, size=(64, 64)):
    """Resize a list of images to specified dimensions."""
    return [cv2.resize(img, size) for img in images]

def load_imgs(folder_path):
    """Load images from specified folder."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images
```

## Requirements

```bash
# Install required packages
pip install opencv-python numpy matplotlib scikit-learn
```

## Usage

```python
# Import required libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess images
folder_path = 'path/to/images'
images = load_imgs(folder_path)
resized_images = resize_images(images)

# Apply K-means clustering
def apply_kmeans(image, k=3):
    # Reshape image to 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # Reconstruct segmented image
    segmented = centers[labels].reshape(image.shape)
    return segmented

# Apply Ratio-Cut clustering
def apply_ratio_cut(image, k=3):
    # Implementation as shown above
    pass

# Example usage
for image in resized_images:
    # K-means segmentation
    kmeans_result = apply_kmeans(image, k=3)
    
    # Ratio-Cut segmentation
    spectral_result = apply_ratio_cut(image, k=3)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(kmeans_result)
    plt.title('K-means')
    plt.subplot(133)
    plt.imshow(spectral_result)
    plt.title('Ratio-Cut')
    plt.show()
```

## Results
<img width="369" alt="image" src="https://github.com/user-attachments/assets/867af3d3-0ed2-4d5e-82a7-72e3b5abb529">

### Sample Visualization
```python
def visualize_results(original, kmeans_result, spectral_result):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(kmeans_result)
    plt.title('K-means Segmentation')
    
    plt.subplot(133)
    plt.imshow(spectral_result)
    plt.title('Ratio-Cut Segmentation')
    
    plt.show()
```

## Author
Sushant Ravva (B21CS084)

## Course Information
Computer Vision (CSL7360)

## Note
This is an academic project focused on implementing and comparing different image segmentation techniques. The implementation is designed for educational purposes and can be extended for more complex applications.
