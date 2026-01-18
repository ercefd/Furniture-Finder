import os
# Fix for macOS threading crash
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import shutil
from PIL import Image
import io
import numpy as np
import torch
from torchvision import transforms

# Import our modules
# Assuming we are running from root or backend is in pythonpath
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import SigLIPEmbedder, FurnitureDetector, ResNetStudent
from faiss_retriever import FaissRetriever
from prompt_generator import PromptGenerator
# from image_editor import ImageEditor
from fastapi.responses import StreamingResponse

app = FastAPI(title="Furniture Prompter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (previous imports)
from fastapi.staticfiles import StaticFiles
import pickle

# ... (StudentWrapper class)

# Paths
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.bin")
PATHS_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paths.pkl")
STUDENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_resnet.pth")
DATASET_DIR = "/Users/ercefiratdemircan/Furniture-Prompter/mudo-images" # Hardcoded for now based on user context

# Mount Static Files
app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")

# Global variables
embedder = None
detector = None
retriever = None
prompt_gen = None
# image_editor = None
image_paths = []

@app.on_event("startup")
async def startup_event():
    global embedder, detector, retriever, prompt_gen, image_paths

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Loading Models on {device}...")
    
    # ... (Embedder loading logic same as before) ...
    if os.path.exists(STUDENT_PATH):
        try:
            embedder = StudentWrapper(STUDENT_PATH, device=device)
            print("Student Model Loaded.")
        except:
            embedder = SigLIPEmbedder(device=device)
    else:
        embedder = SigLIPEmbedder(device=device)

    # Prompt Generator
    print("Loading Prompt Generator...")
    # prompt_gen = PromptGenerator(device=device) # Load on demand or here
    
    # Detector
    print("Loading YOLO Detector...")
    try:
        detector = FurnitureDetector(model_input="yolov8n.pt") # Using Nano for speed
        print("Detector Loaded.")
    except Exception as e:
        print(f"Failed to load Detector: {e}")
        detector = None

    # Index and Paths
    if os.path.exists(INDEX_PATH) and os.path.exists(PATHS_MAP_PATH):
        print("Loading FAISS Index and Paths...")
        retriever = FaissRetriever(INDEX_PATH)
        with open(PATHS_MAP_PATH, "rb") as f:
            image_paths = pickle.load(f)
    else:
        print("Index or Paths not found. Please run build_index_for_api.py")

@app.get("/search_text")
async def search_by_text(q: str):
    if not retriever or not image_paths:
       raise HTTPException(status_code=503, detail="Index not ready")
    
    # Text Embedding
    try:
        if hasattr(embedder, 'get_text_embedding'):
             vector = embedder.get_text_embedding(q)
        else:
             # If using student or old model that doesn't support text
             raise HTTPException(status_code=400, detail="Current model does not support text search.")
    except Exception as e:
        print(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    distances, indices = retriever.search(vector, k=10)
    
    results = []
    for idx, dist in zip(indices, distances):
        if idx < len(image_paths):
            original_path = image_paths[idx]
            rel_path = os.path.basename(original_path)
            url = f"http://localhost:8000/images/{rel_path}"
            results.append({
                "id": int(idx),
                "score": float(dist),
                "image_url": url,
                "name": os.path.basename(original_path)
            })
    return {"results": results}

@app.post("/search")
async def search_furniture(file: UploadFile = File(...)):
    if not retriever or not image_paths:
       raise HTTPException(status_code=503, detail="Index not ready")
       
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Object Detection (Auto-Crop)
    if detector:
        # Detect objects
        boxes = detector.detect(image)
        # Filter for relevant classes (optional, e.g., 'chair', 'couch', 'bed', 'dining table')
        # COCO Classes: 56: chair, 57: couch, 58: potted plant, 59: bed, 60: dining table, 61: toilet...
        # Let's take the one with highest confidence for now
        if len(boxes) > 0:
            # Sort by confidence
            best_box = sorted(boxes, key=lambda x: x.conf[0], reverse=True)[0]
            # Get coordinates
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            # Crop
            print(f"Object Detected: {best_box.cls} Conf: {best_box.conf}. Cropping...")
            image = image.crop((x1, y1, x2, y2))
    
    vector = embedder.get_embedding(image)
    distances, indices = retriever.search(vector, k=10) # Get more candidates
    
    results = []
    for idx, dist in zip(indices, distances):
        if idx < len(image_paths):
            original_path = image_paths[idx]
            # Convert absolute path to relative static URL
            # original: /Users/.../mudo-images/subdir/image.jpg
            # static mount: /images points to /Users/.../mudo-images
            # We need to extract the part relative to mudo-images
            
            # Use basename because stored paths might be relative/broken
            # and we know images are served from DATASET_DIR flatly (or we assume flat for now)
            rel_path = os.path.basename(original_path)
            url = f"http://localhost:8000/images/{rel_path}"
            
            results.append({
                "id": int(idx),
                "score": float(dist),
                "image_url": url,
                "name": os.path.basename(original_path)
            })
        
    return {"results": results}

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    global prompt_gen
    if prompt_gen is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
             device = "mps"
        prompt_gen = PromptGenerator(device=device)
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    prompt = prompt_gen.generate_prompt(image)
    return {"caption": prompt}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
