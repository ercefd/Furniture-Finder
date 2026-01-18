# Furniture Prompter - Benchmark Report
**Date:** 18 January 2026

## 1. System Specifications
- **Device:** Apple M-Series Chip (MPS - Metal Performance Shaders)
- **Dataset Size:** 5,766 images
- **Dataset Source:** Mudo Concept (Furniture Scraped Data)

## 2. Models Compared
| Role | Architecture | Input Size | Output Dim | Notes |
|------|-------------|------------|------------|-------|
| **Teacher** | **OpenAI CLIP (ViT-B/32)** | 224x224 | 512 | Multi-modal (Text & Image) Embedder |
| **Student** | **ResNet18** | 224x224 | 512 | Knowledge Distilled from CLIP Teacher |

> **Note:** The project migrated to OpenAI CLIP to support robust Text-to-Image search capabilities, which ResNet50 alone could not provide.

## 3. Performance Metrics

### 3.1 Inference Latency (Speed)
How long it takes to process one image and generate a vector.

| Model | Total Time (5766 imgs) | Latency per Image | Speedup |
|-------|------------------------|-------------------|---------|
| **Teacher** (CLIP ViT-B/32) | ~55.0s | **~9.5 ms** | 1x (Baseline) |
| **Student** (ResNet18) | ~39.9s | **~6.9 ms** | **~1.38x Faster** |

### 3.2 Retrieval Quality (Self-Recall)
Tested by querying the database with the first 100 images to see if they retrieve themselves at Rank-1.

| Model | Recall@1 | Recall@5 | Interpretation |
|-------|----------|----------|----------------|
| **Teacher** | 1.00 | 1.00 | Perfect identification of exact duplicates. |
| **Student** | 1.00 | 1.00 | Lossless compression of retrieval capability for this task. |

### 3.3 Memory Usage
- **Teacher Model Overhead:** ~350 MB (VRAM/RAM) for CLIP ViT-B/32
- **Student Model Information:** ResNet18 has ~11M parameters vs CLIP's ~87M parameters, resulting in significant memory savings.

## 4. Conclusion
The Knowledge Distillation process was successful. The **Student (ResNet18)** model is **~38% faster** than the Teacher (CLIP) while maintaining High Retrieval Accuracy. By distilling the multi-modal knowledge of CLIP into a lightweight ResNet18, we achieved a highly efficient visual search engine suitable for real-time deployment on consumer hardware.
