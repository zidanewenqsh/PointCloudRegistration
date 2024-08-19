import numpy as np
import cv2
from scipy.signal import convolve2d

def manual_gaussian_blur(image, sigma):
    """ Manually implements a Gaussian blur on an image using a 2D Gaussian kernel.
    
    Args:
    image (numpy array): The input image.
    sigma (float): The standard deviation of the Gaussian kernel.
    
    Returns:
    numpy array: The blurred image.
    """
    # Define the size of the Gaussian kernel based on the sigma
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    
    # Create a 2D Gaussian kernel array
    ax = np.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    # Apply the Gaussian kernel to the image using 2D convolution
    return convolve2d(image, kernel, mode='same')

# Create a sample image: a white square in the middle of a black background
sample_image = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(sample_image, (30, 30), (70, 70), 255, -1)

# Apply Gaussian blur manually
sigma = 5
blurred_manual = manual_gaussian_blur(sample_image, sigma)

# Apply Gaussian blur using OpenCV
blurred_api = cv2.GaussianBlur(sample_image, (0, 0), sigma)

# Display the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(sample_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(132)
plt.imshow(blurred_manual, cmap='gray')
plt.title("Manual Gaussian Blur")
plt.axis("off")

plt.subplot(133)
plt.imshow(blurred_api, cmap='gray')
plt.title("API Gaussian Blur")
plt.axis("off")

plt.show()
