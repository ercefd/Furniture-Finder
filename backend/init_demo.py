import numpy as np
import faiss
import os
import json

def init_demo():
    print("Initializing demo data...")
    
    # 1. Create a dummy FAISS index
    d = 2048 # ResNet50 output dim
    nlist = 10
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)
    
    # Training data
    print("Training dummy index...")
    xt = np.random.rand(1000, d).astype('float32') # 1000 vectors
    index.train(xt)
    
    # Add data
    xb = np.random.rand(100, d).astype('float32') # 100 items in database
    index.add(xb)
    
    faiss.write_index(index, "backend/faiss_index.bin")
    print("backend/faiss_index.bin created.")
    
    # 2. Create metadata? (Optional, API handles it with placeholders)
    print("Done.")

if __name__ == "__main__":
    init_demo()
