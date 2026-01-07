import torch
import numpy as np
import faiss
import os
import pickle
from torch.utils.data import DataLoader
from models import ResNetStudent, SigLIPEmbedder
from distill_siglip import FurnitureDataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

def collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def build_index_for_api(image_dir, output_index="faiss_index.bin", output_paths="paths.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Building index on {device}...")
    
    # Load Model (Use Teacher/CLIP directly for Text Search Compatibility)
    print("Loading CLIP (SigLIPEmbedder wrapper)...")
    embedder = SigLIPEmbedder(device=device)
    
    # Load Data
    # Dataset handles opening images, but Embedder handles transform internally if we pass PIL images
    # But FurnitureDataset returns tensor if transform is passed.
    # Let's set transform=None to get PIL images, then pass to embedder.get_embedding
    
    dataset = FurnitureDataset(image_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn) 
    
    all_embeddings = []
    all_paths = []
    
    print("Generating Embeddings...")
    
    for images, paths in tqdm(dataloader):
        for img, path in zip(images, paths):
             # Processes one by one (slow but safe) or we could batch optimize later
             # embedder.get_embedding expects single PIL image and returns (1, dim) array
             emb = embedder.get_embedding(img)
             all_embeddings.append(emb)
             all_paths.append(path)
             
    # Concatenate
    all_embeddings = np.concatenate(all_embeddings, axis=0) # (N, dim)
    d = all_embeddings.shape[1]
    print(f"Embedding Dimension: {d}")
    
    # Build Index
    print("Building FAISS Index...")
    index = faiss.IndexFlatIP(d) # Inner Product (Cosine Similarity if normalized)
    index.add(all_embeddings)
    
    faiss.write_index(index, output_index)
    with open(output_paths, "wb") as f:
        pickle.dump(all_paths, f)
        
    print(f"Index built! Saved to {output_index}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_index_for_api.py <image_dir> [output_index] [output_paths]")
        if os.path.exists("../mudo-images"):
             build_index_for_api("../mudo-images", "faiss_index.bin", "paths.pkl")
    else:
        img_dir = sys.argv[1]
        out_idx = sys.argv[2] if len(sys.argv) > 2 else "faiss_index.bin"
        out_paths = sys.argv[3] if len(sys.argv) > 3 else "paths.pkl"
        build_index_for_api(img_dir, out_idx, out_paths)
