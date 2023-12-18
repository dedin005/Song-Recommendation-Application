import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
from joblib import dump

# Directory where the encoded images are saved
encoded_images_path = '../../data/encoded_images/'

# Initialize IncrementalPCA with a large number of components
initial_components = 20
ipca = IncrementalPCA(n_components=initial_components)

# Function to load embeddings in batches and fit IncrementalPCA
def fit_incremental_pca(ipca, files, batch_size):
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_data = [np.load(os.path.join(encoded_images_path, file)) for file in batch_files]
        batch_data = np.vstack(batch_data)
        ipca.partial_fit(batch_data)
        print(f"Fitted batch {i//batch_size + 1}/{len(files)//batch_size} for PCA.")
    return ipca

# Load all .npy files from the encoded_images directory
encoded_files = [f for f in os.listdir(encoded_images_path) if f.endswith('.npy')]
batch_size = 800

# Fit the IncrementalPCA on the encoded images
ipca = fit_incremental_pca(ipca, encoded_files, batch_size)

# Determine the number of components to keep 95% variance
cumulative_variance_ratio = np.cumsum(ipca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components_95}")

# Save the IncrementalPCA model
pca_model_path = '../../models/recommender/ipca_model.joblib'
dump(ipca, pca_model_path)
print(f"PCA model saved to {pca_model_path}")

# Directory to save the reduced dimensionality data
reduced_data_path = '../../data/reduced_images/'
os.makedirs(reduced_data_path, exist_ok=True)

# Function to transform embeddings using the fitted IncrementalPCA and save the reduced data
def transform_and_save(ipca, files, batch_size, output_folder):
    ipca.n_components = n_components_95
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_data = [np.load(os.path.join(encoded_images_path, file)) for file in batch_files]
        batch_data = np.vstack(batch_data)
        transformed_data = ipca.transform(batch_data)
        for j, transformed_image in enumerate(transformed_data):
            file_name = f"reduced_image_{i+j}.npy"
            file_path = os.path.join(output_folder, file_name)
            np.save(file_path, transformed_image)
        print(f"Transformed and saved batch {i//batch_size + 1}/{len(files)//batch_size}.")

# Transform the encoded images using the fitted IncrementalPCA and save
transform_and_save(ipca, encoded_files, batch_size, reduced_data_path)
print("All images have been transformed and saved.")
