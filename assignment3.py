from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.svm import SVC
from scipy.spatial.distance import minkowski, cosine
from sklearn.preprocessing import normalize

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data['images'] 
y = data['target']  

# Flatten images for model input 
X_flat = X.reshape(X.shape[0], -1)

# Step 2: Stratified sampling for train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_flat, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Step 3: K-Fold Cross Validation with SVC

'''
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
'''
kf = StratifiedKFold(n_splits=5)
clf = SVC(kernel='linear', C=0.1, gamma='scale')
fold_accuracies = []
for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    clf.fit(X_train_fold, y_train_fold)
    val_preds = clf.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, val_preds)
    fold_accuracies.append(accuracy)

print(f"Mean accuracy across folds: {np.mean(fold_accuracies)}")

# Function to compute clusters and silhouette score
def perform_clustering(X, n_clusters, affinity, linkage='ward'):
    if affinity == 'minkowski' or affinity == 'cosine':
        linkage = 'complete'  # or 'average' or 'single'
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, linkage=linkage)

    cluster_labels = clustering.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    return cluster_labels, score

# Normalize the data for cosine similarity
X_norm = normalize(X_train)

n_clusters = 40
# Step 4(a): Using Euclidean Distance 
labels_euclidean, silhouette_euclidean = perform_clustering(X_train, n_clusters, 'euclidean')
print(f'Silhouette Score (Euclidean): {silhouette_euclidean}')

# Step 4(b): Using Minkowski Distance 
labels_minkowski, silhouette_minkowski = perform_clustering(X_train, n_clusters, 'minkowski')
print(f'Silhouette Score (Minkowski): {silhouette_minkowski}')

# Step 4(c): Using Cosine Similarity 
labels_cosine, silhouette_cosine = perform_clustering(X_norm, n_clusters, 'cosine')
print(f'Silhouette Score (Cosine): {silhouette_cosine}')

cluster_labels = labels_euclidean  

'''
Step 5: Disccusion 

-All three metrics yield relatively low silhouette scores

-Also, The discrepancies between the three distance metrics are relatively small, 

-these distance metrics is not optimal for the Olivetti dataset. 

-This is understandable given that facial recognition tasks often benefit 

'''

# Train a classifier using the cluster labels
fold_accuracies_clustered = []
for train_idx, val_idx in kf.split(X_train, cluster_labels):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = cluster_labels[train_idx], cluster_labels[val_idx]
    
    clf.fit(X_train_fold, y_train_fold)
    val_preds = clf.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, val_preds)
    fold_accuracies_clustered.append(accuracy)

print(f"Mean accuracy across folds with clustering: {np.mean(fold_accuracies_clustered)}")

# Final evaluation on test set
test_preds = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test accuracy: {test_accuracy}")

