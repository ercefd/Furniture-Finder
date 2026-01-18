# Furniture Prompter 🛋️✨

> **Project Overview**
>
> **Furniture Prompter** is an intelligent visual search engine designed to revolutionize how we discover furniture. Bridging the gap between text and vision, it enables users to search for products using both **natural language** ("modern beige armchair with wooden legs") and **images** (uploading a photo).
>
> Powered by advanced **Knowledge Distillation** (compressing large AI models for speed) and **Product Quantization** (optimizing memory), this system demonstrates how to deploy high-performance AI retrieval on consumer hardware.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| 🔍 **Visual Search** | Upload any image to find the most similar furniture pieces from our catalog. |
| 💬 **Text Search** | Find items using descriptive natural language queries. |
| ⚡ **Fast Inference** | Optimized with FAISS, delivering search results in milliseconds. |
| 🤖 **AI Captioning** | Automatically generates detailed descriptions for uploaded images to explain the search context. |
| 🎓 **Student-Teacher AI** | Uses a distilled ResNet18 model that is **~38% faster** and significantly smaller than its teacher (**OpenAI CLIP**), effectively running on standard CPUs/MPS. |

---

## 🛠️ Quick Start Guide

Follow these steps to get the system up and running in minutes.

### 1️⃣ Backend Setup (Python API)
The backend handles the AI models and search logic. It runs on port **8000**.

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Step A: Build the Search Index (Run once)
# This processes the images and creates the 'faiss_index.bin' file.
python build_index_for_api.py ../mudo-images

# Step B: Start the API Server
python api.py
```
✅ **Success:** Server running at `http://localhost:8000`

### 2️⃣ Frontend Setup (React App)
The frontend provides the user interface. It runs on port **3000**.

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
✅ **Success:** App running at `http://localhost:3000` (or check terminal for port)

---

## 📂 Project Architecture

```
Furniture-Prompter/
├── backend/               # 🧠 Brain of the system
│   ├── api.py             # FastAPI Server Entrypoint
│   ├── models.py          # AI Model Definitions (Teacher, Student, CLIP)
│   ├── faiss_retriever.py # Vector Search Logic
│   └── distill_siglip.py  # Knowledge Distillation Training Script
├── frontend/              # 🎨 Face of the system
│   └── src/               # React Components & Pages
├── mudo-images/           # 🗄️ Dataset (Furniture Catalog)
└── BENCHMARK_REPORT.md    # 📊 Scientific Report & Evaluation
```

## 📊 Performance & Evaluation
This project includes a comprehensive evaluation of the Knowledge Distillation process.
- **Speedup:** The Student model is **1.36x faster** than the Teacher.
- **Accuracy:** Maintains **100% Retrieval Recall** for exact matches.
- **Details:** Read the full [Benchmark Report](BENCHMARK_REPORT.md).

## 📝 Notes
- **Device Support:** Optimized for Apple Silicon (MPS). Automatically falls back to CUDA or CPU if unavailable.
- **Dataset:** Contains ~5,700 crawled furniture images. ensure `mudo-images` folder exists in the root.

---
*CENG 543 - Graduate Term Project - Fall 2025*
