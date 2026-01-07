import time
import torch
import numpy as np
import os
import sys
import tracemalloc
import psutil
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

# Import from our modules
from models import SigLIPEmbedder, ResNetStudent
from distill_siglip import FurnitureDataset

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def compute_recall_at_k(index, query_vectors, ground_truth_ids, k=5):
    # Search
    distances, indices = index.search(query_vectors, k)
    
    correct = 0
    total = len(query_vectors)
    
    for i, rows in enumerate(indices):
        # ground_truth_ids[i] is the index of the query itself (self-retrieval)
        # or the class id. 
        # For this benchmark, let's assume "Self-Retrieval" task (Find the exact same image) 
        # as a proxy for robust retrieval if we don't have labeled classes.
        # Or if we have labels, use them.
        # Let's rely on indices matching the input index (0..N).
        if ground_truth_ids[i] in rows:
            correct += 1
            
    return correct / total

def collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def run_benchmark(image_dir, model_path="student_resnet.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    print(f"--- Benchmarking on Dataset: {image_dir} ---")
    
    # 1. Load Data
    # For benchmarking, we want the raw images
    transform = None 
    dataset = FurnitureDataset(image_dir, transform=None)
    if len(dataset) == 0:
        print("No images found. Exiting.")
        return
        
    print(f"Dataset Size: {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 2. Teacher Benchmark
    print("\n[Teacher: SigLIP/ResNet50]")
    tracemalloc.start()
    start_mem = get_process_memory()
    
    teacher = SigLIPEmbedder(device=device)
    
    current_mem = get_process_memory()
    print(f"Teacher Model Memory: {current_mem - start_mem:.2f} MB")
    
    # Teacher Inference Latency
    print("Computing Teacher Embeddings...")
    teacher_embeddings = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Teacher Inference"):
             # Batch process (manually as in distill)
             inputs = []
             for img in images:
                 inputs.append(teacher.preprocess(img))
             inputs = torch.stack(inputs).to(device)
             
             feats = teacher.model(inputs)
             feats = feats / feats.norm(dim=-1, keepdim=True)
             teacher_embeddings.append(feats.cpu().numpy())
             
    total_time = time.time() - start_time
    teacher_embeddings = np.concatenate(teacher_embeddings)
    print(f"Teacher Inference Time: {total_time:.2f}s")
    print(f"Teacher Latency per Image: {(total_time / len(dataset))*1000:.2f} ms")
    
    # 3. Student Benchmark
    print("\n[Student: Distilled ResNet]")
    try:
        # Determine dim from teacher output
        dim = teacher_embeddings.shape[1]
        student = ResNetStudent(output_dim=dim).to(device)
        if os.path.exists(model_path):
            student.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded student weights from {model_path}")
        else:
            print("No student weights found, using random init (Benchmarking architecture only)")
            
        student.eval()
        
        # Measure Memory
        # (This is tricky because teacher is still loaded. We should account for diff or unload teacher)
        # For simplicity, we just measure 'current' total, but ideally we'd isolate.
        
        # Student Inference
        print("Computing Student Embeddings...")
        from torchvision import transforms
        bs_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Re-create dataloader with tensor transform for student efficiency
        # (We can't reuse valid dataloader because it yielded paths)
        ds_student = FurnitureDataset(image_dir, transform=bs_transform)
        dl_student = DataLoader(ds_student, batch_size=32, shuffle=False)
        
        student_embeddings = []
        start_time = time.time()
        with torch.no_grad():
            for images, _ in tqdm(dl_student, desc="Student Inference"):
                images = images.to(device)
                feats = student(images)
                student_embeddings.append(feats.cpu().numpy())
                
        total_time = time.time() - start_time
        student_embeddings = np.concatenate(student_embeddings)
        print(f"Student Inference Time: {total_time:.2f}s")
        print(f"Student Latency per Image: {(total_time / len(dataset))*1000:.2f} ms")
        
    except Exception as e:
        print(f"Student Benchmark Failed: {e}")
        student_embeddings = None

    # 4. Retrieval Quality (Self-Recall@1, @5)
    print("\n--- Retrieval Quality (Self-Recall) ---")
    
    def eval_quality(embeddings, name):
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d) # Inner Product/Cosine since normalized
        index.add(embeddings)
        
        # Query with same embeddings (Self-Retrieval)
        # Ideally, we query with *augmented* versions to test robustness, 
        # but for basic sanity check:
        # If I query with Image A, do I get Image A back at rank 1?
        
        # To make it harder, let's query with the first 100 images
        k = 5
        queries = embeddings[:100]
        gt_ids = np.arange(100)
        
        r1 = compute_recall_at_k(index, queries, gt_ids, k=1)
        r5 = compute_recall_at_k(index, queries, gt_ids, k=5)
        
        print(f"[{name}] Self-Recall@1: {r1:.2f}")
        print(f"[{name}] Self-Recall@5: {r5:.2f}")
        
    eval_quality(teacher_embeddings, "Teacher")
    if student_embeddings is not None:
        eval_quality(student_embeddings, "Student")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <image_dir> [student_model_path]")
    else:
        img_dir = sys.argv[1]
        model_p = sys.argv[2] if len(sys.argv) > 2 else "student_resnet.pth"
        run_benchmark(img_dir, model_p)
