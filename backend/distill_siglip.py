import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import os
import glob
from models import SigLIPEmbedder, ResNetStudent
import time

class FurnitureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image
        return image_tensor, path

def train_student(image_dir, output_model_path="student_resnet.pth", epochs=5, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Teacher
    print("Loading Teacher...")
    teacher = SigLIPEmbedder(device=device)
    
    # Check Teacher Dimension dynamically
    # Create a dummy input to check output dim
    dummy_img = Image.new("RGB", (224, 224))
    dummy_out = teacher.get_embedding(dummy_img)
    teacher_dim = dummy_out.shape[1]
    print(f"Teacher output dimension: {teacher_dim}")

    # Student
    print(f"Initializing Student (dim={teacher_dim})...")
    student = ResNetStudent(output_dim=teacher_dim).to(device)
    student.train()

    # Data
    # Student Transform
    student_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Simplistic resize for student
        transforms.RandomHorizontalFlip(), # Add some augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FurnitureDataset(image_dir, transform=student_transform)
    if len(dataset) == 0:
        print("No images found in", image_dir)
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # Distillation loss

    print(f"Starting training for {epochs} epochs on {len(dataset)} images...")
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            
            # --- Teacher Step ---
            # Process batch for teacher
            # We need to reload/process images for teacher because it might have different transforms
            # and we want clean targets.
            # Ideally, we should do this in __getitem__ but for now we do it here (slower but safe)
            with torch.no_grad():
                # Preprocess for teacher
                teacher_inputs = []
                for p in paths:
                    img = Image.open(p).convert("RGB")
                    # Use teacher's internal preprocess
                    # SigLIPEmbedder.preprocess expects PIL and returns Tensor
                    teacher_inputs.append(teacher.preprocess(img))
                
                teacher_inputs = torch.stack(teacher_inputs).to(device)
                
                # Forward pass through teacher model
                # We access internal model to process batch
                teacher_features = teacher.model(teacher_inputs)
                
                # Normalize (crucial for Cosine Similarity / Embedding matching)
                target = teacher_features / teacher_features.norm(dim=-1, keepdim=True)

            # --- Student Step ---
            optimizer.zero_grad()
            student_embeds = student(images)
            
            # Loss
            loss = criterion(student_embeds, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} Complete. Avg Loss: {avg_loss:.4f}. Time: {time.time()-start_time:.1f}s")

    torch.save(student.state_dict(), output_model_path)
    print(f"Student model saved to {output_model_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python distill_siglip.py <image_dir>")
    else:
        train_student(sys.argv[1])
