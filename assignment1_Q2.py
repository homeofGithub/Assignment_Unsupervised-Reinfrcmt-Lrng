# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:37:17 2024
Yi Chen ID: 300952167
@author: lette
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Task 1: Generate Swiss roll dataset
X, y = make_swiss_roll(n_samples=1000, noise=0.2)

# Convert continuous labels to discrete classes
y = np.digitize(y, bins=np.linspace(min(y), max(y), 4))

# Task 2: Plot the resulting generated Swiss roll dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
plt.title('Swiss Roll Dataset')
plt.show()

# Task 3: Apply Kernel PCA with different kernels
kernels = ['linear', 'rbf', 'sigmoid']
kpca_results = {}

for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X)
    kpca_results[kernel] = X_kpca

    # Task 4: Plot the kPCA results
    plt.figure()
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(f'Kernel PCA with {kernel} kernel')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Task 5: Using kPCA and a kernel, and apply Logistic Regression for classification
pipeline = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': np.linspace(0.03, 0.05, 10)
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X, y)

print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Task 6: Plot the results from using GridSearchCV
best_kpca = grid_search.best_estimator_.named_steps['kpca']
X_best_kpca = best_kpca.transform(X)

plt.figure()
plt.scatter(X_best_kpca[:, 0], X_best_kpca[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Best kPCA results with GridSearchCV')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
