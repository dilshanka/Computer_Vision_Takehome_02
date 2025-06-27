import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


input_path = "data/input_images/input_task2.png"
output_path = "data/output_images/task2/region-growing.png"


def region_growing(image, seed, threshold=20):
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)

    stack = [seed]
    seed_intensity = image[seed]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue

        visited[x, y] = True
        segmented[x, y] = image[x, y]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if abs(int(image[nx, ny]) - int(seed_intensity)) <= threshold:
                    stack.append((nx, ny))

    return segmented


# Load image in grayscale
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not read {input_path}")


# Use Otsu's threshold to separate foreground
hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
p = hist / hist.sum()
omega = np.cumsum(p)
mu = np.cumsum(p * np.arange(256))
mu_T = mu[-1]
sigma_b2 = (mu_T * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
thr = np.argmax(sigma_b2)

foreground = image > thr
labels, nlab = ndi.label(foreground)
sizes = ndi.sum(foreground, labels, index=np.arange(1, nlab + 1))
largest_lbl = np.argmax(sizes) + 1
seed_row, seed_col = map(int, ndi.center_of_mass(foreground, labels, largest_lbl))
seed_point = (seed_row, seed_col)


# Apply Region Growing

segmented_image = region_growing(image, seed_point, threshold=20)

# Save segmented image
cv2.imwrite(output_path, segmented_image)


# Display Results

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Seed Point Marked")
plt.imshow(image, cmap="gray")
plt.scatter(seed_col, seed_row, c="red", s=40, marker="x")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmented Region")
plt.imshow(segmented_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Segmented image saved to: {output_path}")
