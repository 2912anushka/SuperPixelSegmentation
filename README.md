# Image Segmentation using K-means Clustering

This repository contains code for performing image segmentation using K-means clustering. The code utilizes the OpenCV and scikit-learn libraries to load and process the image, perform clustering, and display the segmented image.

## Dependencies

The following dependencies are required to run the code:

- Python 3.x
- OpenCV
- scikit-learn
- matplotlib
- numpy

You can install the dependencies using pip:

```
pip install opencv-python scikit-learn matplotlib numpy
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/image-segmentation.git
```

2. Change to the repository directory:

```bash
cd SuperPixelSegmentation
```

3. Place your input image in the repository directory.

4. Run the script:

```bash
python algo.py
```

5. Follow the instructions and enter the number of clusters (k) for image segmentation.

6. The script will display the original image and the segmented image with the specified number of clusters. The segmented image will be saved in the same directory as "segmented_image.jpg".

## Results

The script performs K-means clustering on the input image and generates a segmented image based on the specified number of clusters. The segmented image represents different regions of the image assigned to different clusters. The resulting image can be used for various applications such as object recognition, image compression, and image editing.

