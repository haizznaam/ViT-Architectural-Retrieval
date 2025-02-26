import os
import pickle
import numpy as np
from feature_extract_model import ImageRetrievalModel

# Define query image and K nearest images to retrieve
query_image = "test-3.png"

# Modify K as needed
K = 4

# Initialize the model
model = ImageRetrievalModel(data_folder=".data/", save_folder=".save/")

# Ensure embeddings exist
model.ensure_embeddings()

# Get K most similar images
nearest_images = model.get_K_similar_images(query_image, K=K, visualize=True)

# Print the results
for img_path, similarity in nearest_images:
    print(f"Image: {img_path}, Similarity: {similarity:.4f}")
