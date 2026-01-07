import torch
import torch.nn as nn
from transformers import SiglipProcessor, SiglipModel
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        print(f"Loading CLIP Model: {model_name} on {device}...")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.cpu().numpy()
    
    def get_text_embedding(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.cpu().numpy()

# Legacy Wrapper for compatibility if needed, but we will switch to CLIP
class SigLIPEmbedder:
    def __init__(self, model_name="google/siglip-base-patch16-224", device="cpu"):
        # We are swiching to CLIP for stability and Text Search support
        print("Switched from SigLIP to CLIP for Text-Search capability.")
        self.delegate = CLIPEmbedder(device=device)
        self.model = self.delegate.model # Hack for access if needed

    def get_embedding(self, image: Image.Image):
        return self.delegate.get_embedding(image)
    
    def get_text_embedding(self, text: str):
        return self.delegate.get_text_embedding(text)


class ResNetStudent(nn.Module):
    def __init__(self, output_dim=768, pretrained=True):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        # Remove fc layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
         # L2 normalize
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

class FurnitureDetector:
    def __init__(self, model_input="yolov8n.pt"): 
        # Using nano model for speed in demo
        self.model = YOLO(model_input)
    
    def detect(self, image: Image.Image):
        results = self.model(image)
        # Return boxes: [x1, y1, x2, y2, conf, cls]
        return results[0].boxes

class Segmenter:
    def __init__(self, model_type="sam_b.pt"): # Using SAM base
        # Ultralytics supports SAM models. 
        # Check if "sam2_b.pt" is available or fallback to "sam_b.pt"
        self.model = SAM(model_type)

    def segment(self, image_path, bboxes=None):
        # Segment based on bboxes if provided, else full image
        if bboxes is not None:
            results = self.model(image_path, bboxes=bboxes)
        else:
            results = self.model(image_path)
        return results
