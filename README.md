
# Image Processing Project

This project implements **image segmentation** techniques using Python. The main tasks include adding **Gaussian noise** to an image, applying **Otsu’s thresholding**, and performing **region-growing segmentation**. The region-growing technique is based on starting from seed points inside the object of interest and expanding the region by including neighboring pixels that are similar in pixel value.


## Description

This project demonstrates the following image processing techniques:

1. **Gaussian Noise Addition & Otsu's Thresholding**:
   - Adds Gaussian noise to a synthetic image.
   - Applies Otsu's thresholding to automatically segment the image into foreground and background.

2. **Region Growing Segmentation**:
   - Implements the region-growing algorithm starting from a seed point inside the object of interest (foreground).
   - Expands the region by recursively adding neighboring pixels that have pixel values within a pre-defined threshold.

## Requirements

You need to install the following Python libraries:

- **NumPy**: For numerical operations and image manipulation.
- **OpenCV**: For image processing functions (such as thresholding and noise addition).
- **Matplotlib**: For displaying images.

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
opencv-python
matplotlib
```

## Folder Structure

Here is the recommended folder structure for this project:

```
image_processing_project/
├── data/
│   ├── input_images/        
│   └── output_images/       
├── src/
│   └── region_growing_segmentation.py  
│   └── gaussian_noise_otsu_thresholding.py  
├── requirements.txt         
└── README.md                
```

## How to Run

1. **Prepare the Image**:
   - Place the input image you want to process in the `data/input_images/` folder.
   - The image should be in grayscale format, with distinct pixel values for the objects and background.

2. **Running the Gaussian Noise & Otsu’s Thresholding Script**:
   - Open a terminal and navigate to the root directory of the project.
   - Run the script that adds Gaussian noise and applies Otsu’s thresholding:

     ```bash
     python src/gaussian_noise_otsu_thresholding.py
     ```

   - This will process the image, apply Gaussian noise, and then apply Otsu's thresholding to segment the image.
   - The processed images will be saved in the `data/output_images/` folder.

3. **Running the Region-Growing Segmentation Script**:
   - Run the script for **region-growing segmentation** to perform the segmentation starting from a seed point:

     ```bash
     python src/region_growing_segmentation.py
     ```

   - The output segmented image will be saved to the `data/output_images/` folder.

4. **Displaying the Results**:
   - Both scripts display the following images using Matplotlib:
     - The **original image**.
     - The **noisy image** (after adding Gaussian noise).
     - The **segmented image** (after applying Otsu’s thresholding or region-growing segmentation).


