import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


input_image_path = "data/input_images/input_task1.png"
output_noisy_image_path = "data/output_images/task1/noisy_image.png"
output_otsu_image_path = "data/output_images/task1/otsu_thresholded_image.png"


image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)


# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)


# Apply Otsu's thresholding
def otsu_thresholding(image):
    _, thresholded_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded_image


# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image)

# Apply Otsu's Thresholding to the noisy image
otsu_image = otsu_thresholding(noisy_image)

# Save the noisy image and the Otsu thresholded image
cv2.imwrite(output_noisy_image_path, noisy_image)
cv2.imwrite(output_otsu_image_path, otsu_image)

# Display the images
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

# Noisy Image
plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

# Otsu's Thresholding Image
plt.subplot(1, 3, 3)
plt.title("Otsu's Thresholding")
plt.imshow(otsu_image, cmap="gray")
plt.axis("off")

plt.show()
