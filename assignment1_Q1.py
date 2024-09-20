# -*- coding: utf-8 -*-
"""
Spyder Editor
Yi Chen ID: 300952167
This is a temporary script file.
"""


import arff
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

# Path to the mnist_784.arff file
file_path = "C:/Users/lette/OneDrive/Desktop/Unsupervised & Reinfrcmt/Assignmen1/mnist_784.arff"

# Load the ARFF file
with open(file_path, 'r') as file:
    mnist_data = arff.load(file)

# Convert loaded data into numpy array for manipulation
data_array = np.array(mnist_data['data'])


X = data_array[:, :-1].astype(float)
y = data_array[:, -1].astype(int) 

# Task 2: Display each digit 
for i in range(10):
    
    indices_of_i = np.where(y == i)[0][:5]  
    
    # Plot each instance of digit i
    for count, index in enumerate(indices_of_i):
        plt.subplot(1, 5, count + 1)
        plt.imshow(X[index].reshape(28, 28), cmap='gray')
        plt.title(f"Digit {i}")
        plt.axis('off')
    
    plt.show()

# Task 3: PCA to retrieve the 1st and 2nd principal components and output their explained variance ratio
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio of the 1st and 2nd principal components:", explained_variance_ratio)

# Task 4: Plot the projections of the 1st and 2nd principal components onto a 1D hyperplane
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=1)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('Projections of the 1st and 2nd Principal Components')
plt.colorbar()
plt.show()

# Task 5: Use Incremental PCA to reduce the MNIST dataset down to 154 dimensions
ipca = IncrementalPCA(n_components=154, batch_size=200)
X_ipca = ipca.fit_transform(X)

# Task 6: Display the original and compressed images from (5)
# Display original images
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Original {i}")
    plt.axis('off')

# Display compressed images
X_ipca_reconstructed = ipca.inverse_transform(X_ipca)
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(X_ipca_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title(f"Compressed {i}")
    plt.axis('off')

plt.show()

