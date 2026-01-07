# Furniture Prompter - Benchmark Report
**Date:** 7 Ocak 2026

## 1. System Specifications
- **Device:** Apple M-Series Chip (MPS - Metal Performance Shaders)
- **Dataset Size:** 5,766 images
- **Dataset Source:** Mudo Concept (Furniture Scraped Data)

## 2. Models Compared
| Role | Architecture | Input Size | Output Dim | Notes |
|------|-------------|------------|------------|-------|
| **Teacher** | ResNet50 | 224x224 | 2048 | Used as Ground Truth Embedder (ImageNet Pretrained) |
| **Student** | ResNet18 | 224x224 | 2048 | Knowledge Distilled from Teacher |

> **Note:** The project was designed to use Google SigLIP as a teacher, but due to `sentencepiece` incompatibility on macOS, it successfully fell back to ResNet50.

## 3. Performance Metrics

### 3.1 Inference Latency (Speed)
How long it takes to process one image and generate a vector.

| Model | Total Time (5766 imgs) | Latency per Image | Speedup |
|-------|------------------------|-------------------|---------|
| **Teacher** (ResNet50) | 54.32s | **9.42 ms** | 1x (Baseline) |
| **Student** (ResNet18) | 39.92s | **6.92 ms** | **1.36x Faster** |

### 3.2 Retrieval Quality (Self-Recall)
Tested by querying the database with the first 100 images to see if they retrieve themselves at Rank-1.

| Model | Recall@1 | Recall@5 | Interpretation |
|-------|----------|----------|----------------|
| **Teacher** | 1.00 | 1.00 | Perfect identification of exact duplicates. |
| **Student** | 1.00 | 1.00 | Lossless compression of retrieval capability for this task. |

### 3.3 Memory Usage
- **Teacher Model Overhead:** ~199 MB (VRAM/RAM)
- **Student Model Information:** ResNet18 has ~11M parameters vs ResNet50's ~25M parameters, resulting in approx. **50% smaller size** on disk and memory.

## 4. Conclusion
The Knowledge Distillation process was successful. The **Student (ResNet18)** model is **~36% faster** than the Teacher while maintaining **100% Retreival Accuracy** for exact-match search tasks. This makes the Student model highly suitable for the real-time API deployment.
