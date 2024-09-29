
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data['images'] 
y = data['target']  

# Flatten images for model input 
X_flat = X.reshape(X.shape[0], -1)

# step 2 80% training, 10% validation, 10% test with stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X_flat, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Step 3: K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5)
clf = SVC(kernel='linear')

# K-Fold Cross Validation
fold_accuracies = []
for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    clf.fit(X_train_fold, y_train_fold)
    val_preds = clf.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, val_preds)
    fold_accuracies.append(accuracy)

print(f"Mean accuracy across folds: {np.mean(fold_accuracies)}")

# Step 4: Use K-Means to reduce the dimensionality of the set
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_train)

# Test multiple K values for KMeans

best_k = 0
best_score = -1
for k in range(2, 20):  # Explore different k values
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    
    if score > best_score:
        best_k = k
        best_score = score

print(f"Best K: {best_k}, Best Silhouette Score: {best_score}")

# Step 5: train a classifier in step (3)
kmeans = KMeans(n_clusters=best_k, random_state=42)
train_clusters = kmeans.fit_predict(X_pca)

clf.fit(X_pca, train_clusters)
cluster_preds = clf.predict(pca.transform(X_val))
accuracy = accuracy_score(kmeans.predict(pca.transform(X_val)), cluster_preds)

print(f"Cluster classification accuracy: {accuracy}")

# Step 6: DBSCAN Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Count the number of clusters
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters found by DBSCAN: {n_clusters}")
