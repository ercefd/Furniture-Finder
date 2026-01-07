import faiss
import numpy as np
import os
import pickle

def build_index(embeddings_path, index_output_path="faiss_index.bin", nlist=100, m=8, nbits=8):
    # Load embeddings
    # Assuming embeddings are stored as a numpy array in a .npy file
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file {embeddings_path} not found.")
        return

    embeddings = np.load(embeddings_path)
    d = embeddings.shape[1]
    
    # IVF-PQ
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    
    # Train
    print("Training FAISS index...")
    index.train(embeddings)
    
    # Add
    print("Adding vectors to index...")
    index.add(embeddings)
    
    # Save
    faiss.write_index(index, index_output_path)
    print(f"Index saved to {index_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_faiss_index.py <embeddings.npy>")
    else:
        build_index(sys.argv[1])
