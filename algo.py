import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('peacock.jpeg')

# Convert BGR to RGB for proper visualization
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(rgb)
plt.axis('off')

# Reshape and prepare data for clustering
p_val = rgb.reshape((-1, 3))
p_val = np.float32(p_val)

# Get the number of clusters from user input
k = int(input("Enter the number of clusters (k): "))

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(p_val)

# Get the labels and cluster centers
labels = kmeans.predict(p_val)
centers = kmeans.cluster_centers_
centers = np.uint8(centers)

# Reshape the image based on the labels
img = centers[labels].reshape(rgb.shape)

# Display the segmented image
plt.subplot(1, 2, 2)
plt.title(f'Segmented Image (k={k})')
plt.imshow(img)
plt.axis('off')

# Show the figure with the original and segmented images
plt.tight_layout()
plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('peacock.jpeg')

# Convert BGR to RGB for proper visualization
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(rgb)
plt.axis('off')

# Reshape and prepare data for clustering
p_val = rgb.reshape((-1, 3))
p_val = np.float32(p_val)

# Get the number of clusters from user input
k = int(input("Enter the number of clusters (k): "))

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(p_val)

# Get the labels and cluster centers
labels = kmeans.predict(p_val)
centers = kmeans.cluster_centers_
centers = np.uint8(centers)

# Calculate the percentage of segmentation achieved for each cluster
segmentation_percentages = []
total_pixels = p_val.shape[0]
for cluster in range(k):
    cluster_pixels = np.sum(labels == cluster)
    percentage = (cluster_pixels / total_pixels) * 100
    segmentation_percentages.append(percentage)

# Reshape the image based on the labels
img = centers[labels].reshape(rgb.shape)

# Display the segmented image
plt.subplot(1, 2, 2)
plt.title(f'Segmented Image (k={k})')
plt.imshow(img)
plt.axis('off')

# Show the figure with the original and segmented images
plt.tight_layout()
plt.show()

# Plot the percentage of segmentation achieved for each cluster
plt.figure(figsize=(8, 4))
plt.bar(range(k), segmentation_percentages, color='blue')
plt.xlabel('Cluster')
plt.ylabel('Segmentation Percentage')
plt.title('Segmentation Percentage per Cluster')
plt.xticks(range(k))
plt.show()

