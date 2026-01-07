import faiss
import numpy as np
import torch

class FaissRetriever:
    def __init__(self, index_path):
        self.index = faiss.read_index(index_path)
        
    def search(self, query_vector, k=5):
        # query_vector should be (1, d) numpy array
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]
