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
import argparse

class FurnitureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = image
            return image_tensor, path
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy image or handle gracefully
            # For simplicity, returning None and filtering in collate could be better, 
            # but let's just return a zero tensor if transform is expected
            if self.transform:
                return torch.zeros((3, 224, 224)), path
            return Image.new("RGB", (224, 224)), path

def train_student(train_dir, val_dir=None, test_dir=None, output_model_path="student_resnet.pth", epochs=5, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Teacher
    print("Loading Teacher...")
    teacher = SigLIPEmbedder(device=device)
    
    # Check Teacher Dimension dynamically
    dummy_img = Image.new("RGB", (224, 224))
    dummy_out = teacher.get_embedding(dummy_img)
    teacher_dim = dummy_out.shape[1]
    print(f"Teacher output dimension: {teacher_dim}")

    # Student
    print(f"Initializing Student (dim={teacher_dim})...")
    student = ResNetStudent(output_dim=teacher_dim).to(device)
    student.train()

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    print(f"Loading Training Data from: {train_dir}")
    train_dataset = FurnitureDataset(train_dir, transform=train_transform)
    if len(train_dataset) == 0:
        print(f"No images found in training directory: {train_dir}")
        return
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = None
    if val_dir:
        print(f"Loading Validation Data from: {val_dir}")
        val_dataset = FurnitureDataset(val_dir, transform=val_transform)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            print(f"Warning: Validation directory provided but empty: {val_dir}")

    test_loader = None
    if test_dir:
        print(f"Loading Test Data from: {test_dir}")
        test_dataset = FurnitureDataset(test_dir, transform=val_transform) # Use val transform for test
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            print(f"Warning: Test directory provided but empty: {test_dir}")

    
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # Distillation loss

    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (images, paths) in enumerate(train_loader):
            images = images.to(device)
            
            # --- Teacher Step ---
            with torch.no_grad():
                teacher_pil_images = []
                for p in paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        teacher_pil_images.append(img)
                    except:
                        teacher_pil_images.append(Image.new("RGB", (224, 224)))
                
                # Use the delegate's processor (Hugging Face Processor)
                # It handles normalization, resizing, converting to tensor
                # We access it via .delegate.processor as per models.py structure
                processor = teacher.delegate.processor
                inputs = processor(images=teacher_pil_images, return_tensors="pt", padding=True).to(device)
                
                # Get embeddings using the model directly
                # CLIPModel from HF uses get_image_features
                teacher_features = teacher.model.get_image_features(**inputs)
                
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
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation Step ---
        avg_val_loss = 0.0
        if val_loader:
            student.eval()
            val_loss = 0
            with torch.no_grad():
                for images, paths in val_loader:
                    images = images.to(device)
                    
                    # Teacher targets for val
                    teacher_pil_images = []
                    for p in paths:
                        try:
                            img = Image.open(p).convert("RGB")
                            teacher_pil_images.append(img)
                        except:
                             teacher_pil_images.append(Image.new("RGB", (224, 224)))

                    processor = teacher.delegate.processor
                    inputs = processor(images=teacher_pil_images, return_tensors="pt", padding=True).to(device)
                    teacher_features = teacher.model.get_image_features(**inputs)
                    target = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                    
                    student_embeds = student(images)
                    loss = criterion(student_embeds, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} Complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}. Time: {time.time()-start_time:.1f}s")
        else:
            print(f"Epoch {epoch+1}/{epochs} Complete. Train Loss: {avg_train_loss:.4f}. Time: {time.time()-start_time:.1f}s")

    torch.save(student.state_dict(), output_model_path)
    print(f"Student model saved to {output_model_path}")

    # --- Test Step ---
    if test_loader:
        print("\nRunning Evaluation on Test Set...")
        student.eval()
        test_loss = 0
        with torch.no_grad():
            for images, paths in test_loader:
                images = images.to(device)
                
                # Teacher targets
                teacher_pil_images = []
                for p in paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        teacher_pil_images.append(img)
                    except:
                        teacher_pil_images.append(Image.new("RGB", (224, 224)))

                processor = teacher.delegate.processor
                inputs = processor(images=teacher_pil_images, return_tensors="pt", padding=True).to(device)
                teacher_features = teacher.model.get_image_features(**inputs)
                target = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                
                student_embeds = student(images)
                loss = criterion(student_embeds, target)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Set Clean Loss (MSE): {avg_test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill SigLIP to ResNet Student with Train/Val/Test splits")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, help="Path to validation dataset")
    parser.add_argument("--test_dir", type=str, help="Path to test dataset")
    parser.add_argument("--output", type=str, default="student_resnet.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    
    train_student(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
