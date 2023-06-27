# OpenCV Image Matching

This project aims to demonstrate image matching techniques using OpenCV, specifically SIFT (Scale-Invariant Feature Transform), histogram comparison, and template matching.

## Algorithms

- SIFT: This technique detects and describes distinctive keypoints in images, allowing for robust image matching even in the presence of scale, rotation, and viewpoint changes.

- Histogram Comparison: This technique compares the histograms of images to measure their similarity based on pixel intensity distributions.

- Template Matching: This technique searches for a template image within a larger target image by comparing pixel values at different positions.

## Prerequisites

- Python 3.x
- OpenCV library (install using `pip install opencv-python`)

## Installation

1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd opencv-image-matching-project`

## Usage

1. Place your query images in the `query_images` directory.
2. Place the target images in the `target_images` directory.
3. Open the algorithm file and customize the algorithm and parameters according to your requirements.
4. Run the script. 

## Results

The results with matching score for each query image will be printed with best match in database image.
