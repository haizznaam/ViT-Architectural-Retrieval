import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import logging
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class ImageRetrievalModel:
    def __init__(self, data_folder: str, save_folder: str, device: Optional[str] = None):
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.vector_file = os.path.join(save_folder, "vectors.pkl")
        self.path_file = os.path.join(save_folder, "paths.pkl")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")

        # Initialize ViT model
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.forward = self._forward_features
        self.model.eval()
        self.model.to(self.device)

        self.feature_dim = 768  # Fixed for ViT-B/16

        # ViT-specific transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info(f"Initialized ViT feature extractor with dimension: {self.feature_dim}")

    def _forward_features(self, x):
        """Modified forward pass to get embeddings"""
        x = self.model._process_input(x)
        n = x.shape[0]
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.encoder(x)
        return x[:, 0]
    
    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image"""
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(image)
        features = features.cpu().numpy().squeeze()
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features
    
    def compute_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def get_K_similar_images(self, query_image: str, K: int = 4, visualize: bool = True) -> List[Tuple[str, float]]:
        """Find K most similar images using cosine similarity"""
        query_vector = self.extract_features(query_image)
        vector_db, path_db = self.load_embeddings()
        similarities = np.array([self.compute_cosine_similarity(query_vector, vec) for vec in vector_db])
        ids = np.argsort(similarities)[::-1][:max(1, K)]  # Ensure at least 1 result
        nearest_images = [(path_db[i], similarities[i]) for i in ids]
        if visualize:
            self.visualize_results(nearest_images, max(1, K))
        return nearest_images
    
    def visualize_results(self, nearest_images: List[Tuple[str, float]], K: int):
        """Visualize the K most similar images"""
        fig, axes = plt.subplots(1, max(1, K), figsize=(5 * max(1, K), 5))
        if K == 1:
            axes = [axes]  # Ensure axes is iterable for single image case
        for idx, (image_path, similarity) in enumerate(nearest_images):
            img = Image.open(image_path)
            axes[idx].imshow(img)
            # image_name = os.path.basename(image_path)
            image_name = image_path
            axes[idx].set_title(f"{image_name}\nScore: {similarity:.4f}", fontsize=6)
            axes[idx].axis('off')
        plt.tight_layout()
        plt.show()
    
    def ensure_embeddings(self):
        """Check if embeddings exist, otherwise process the image folder"""
        if not (os.path.exists(self.vector_file) and os.path.exists(self.path_file)):
            print("Embedding vectors not found. Embedding full image folder...")
            self.process_images()
    
    def process_images(self):
        """Extract features from all images and save to files"""
        image_vectors = {}
        paths = []
        vectors = []
        for root, _, files in os.walk(self.data_folder):
            for file in tqdm(files, desc="Processing Images"):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(root, file)
                    feature_vector = self.extract_features(img_path)
                    paths.append(img_path)  # Save full image path
                    vectors.append(feature_vector)
                    image_vectors[img_path] = feature_vector.tolist()
        with open(self.vector_file, "wb") as vf, open(self.path_file, "wb") as pf:
            pickle.dump(vectors, vf)
            pickle.dump(paths, pf)
        print(f"Saved {len(paths)} image vectors!")
    
    def load_embeddings(self):
        """Load vector database from files"""
        with open(self.vector_file, "rb") as vf, open(self.path_file, "rb") as pf:
            vectors = pickle.load(vf)
            paths = pickle.load(pf)
        return vectors, paths
