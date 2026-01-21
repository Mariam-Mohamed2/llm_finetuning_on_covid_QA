## **Project Overview**

This project presents an **end-to-end pipeline for fine-tuning and deploying a domain-specific LLM** for **COVID-19 vaccine safety and side effects Question Answering (QA)**.  
It covers the complete workflow starting from raw medical documents (PDFs), passing through structured dataset generation and LoRA fine-tuning, and ending with evaluation and deployment readiness.

The pipeline is designed to be:
- Modular
- Reproducible
- Production-oriented
- Fully traceable to original medical sources

---

## Main Notebooks

This repository contains two core notebooks:

1. **Fine-Tuning Pipeline**
   - `COVID19_Vaccine_QA_FineTuning_Pipeline.ipynb`  
   Implements the full data ingestion → dataset generation → LoRA fine-tuning → evaluation pipeline.

2. **Deployment Notebook**
   - `deployment_fastapi.ipynb`  
   Demonstrates how the fine-tuned model can be deployed using FastAPI.

> ⚠️ Note  
> This repository does not contain all model checkpoints and artifacts due to size limits.  
> Large files are hosted externally on Drive
> You can download them from Google Drive:

**Google Drive:** [Drive Link](https://drive.google.com/drive/folders/1_GzXpxDDuF7W4kQw4wsC-4QCgbmBihNW?usp=drive_link)
After downloading, place the contents inside:


---
# First: Fine-Tuning Notebook overview

## 1. Data Collection & Document Processing

The pipeline starts by processing **COVID-19 related medical sources**, including:
- WHO reports
- Clinical safety studies
- Vaccine side effects analysis
- Long-term risk and biomarker research
- Johnson & Johnson / Pfizer safety studies

Using `pdfplumber`, each PDF is:
- Parsed page-by-page
- Converted into structured text
- Enriched with metadata:

```json
{
  "text": "...",
  "metadata": {
    "source": "WHO_XEC_Variant_Risk.pdf",
    "page": 3,
    "domain": "COVID-19 Vaccine Safety"
  }
}
```


## 2. Structured Output Schemas (Pydantic)

Two different structured output schemas are used in this project to guarantee consistency, reliability, and production-ready outputs.

---

### 1. Model Answer Schema (Inference & Evaluation)  
**`class GenerateAnswer`**

Used for:
- Model inference  
- Model evaluation  
- Deployment responses (FastAPI)  

Purpose:
- Ensures clean, machine-readable medical answers  
- Enforces strict JSON formatting  
- Provides confidence estimation for each response  
- Supports traceability with source and page attribution  

This schema is designed to make the model suitable for real-world medical QA systems.

---

### 2. QA Dataset Generation Schema  
**`class QAPair`**  
**`class PageExtraction`**

Used for:

- Synthetic QA dataset generation
- High-quality supervised fine-tuning
- Medical dataset consistency
- Each page produces 20 grounded QA pairs.

---

## 3. Baseline Model Evaluation

**Base model used:**  
`Qwen/Qwen2.5-1.5B-Instruct`

Before fine-tuning, the base model was evaluated using the structured QA pipeline and Pydantic schemas.

Observations:
- The model shows clear **hallucination tendencies**
- It often produces **generic medical answers**
- It fails to capture **precise, evidence-based details** from COVID-19 vaccine literature
- Responses are not reliably grounded in the provided documents

---

## 4. QA Dataset Generation (Main Highlights)

- Uses **Gemini 2.5 flash lite API** to automatically generate high-quality QA pairs.
- Every generated answer:
  - Is **strictly grounded** in the source text.
  - Includes the **source filename** and **page number**.
  - Is validated using **Pydantic schemas** to enforce structure and correctness.
- Dataset is stored in **JSONL format** for scalability and easy loading.
- Includes **rate-limit handling and retry logic** to ensure robust large-scale generation.
- Produces a **reusable, frozen dataset artifact** that can be shared, versioned, and used for reproducible experiments.

---

## 5. Fine-Tuning with LLaMA-Factory

- **Framework:** LLaMA-Factory  
- **Method:** Supervised Fine-Tuning (SFT)  
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
- Dataset converted into **LLaMA-Factory compatible format**.
- **Train / Validation split** applied for reliable evaluation.
- Fully **YAML-driven configuration**, enabling:
  - Reproducibility  
  - Easy hyperparameter tuning  
  - Clean experiment management  

This setup reflects a real-world production-style fine-tuning workflow.


---

## 6. LoRA Fine-Tuning Results

**Fine-tuning method:**  
- LoRA (Low-Rank Adaptation)  
- Efficient and memory-friendly  
- Only **3 epochs**, proving fast domain adaptation capability  

**Training monitoring (Weights & Biases):**  
All training curves and loss dynamics can be viewed from the run link below:

- 🚀 **Run (recommended for viewing training diagrams):**  
  https://wandb.ai/mariam9-bfcai/llamafactory/runs/rhseelv4  

- ⭐ Project:  
  https://wandb.ai/mariam9-bfcai/llamafactory  

**Final training loss (from W&B):**
```python
{'loss': 0.9536, 'grad_norm': 0.3893, 'learning_rate': 9.23e-05, 'epoch': 1.2}
```

---
# 2. Deployment Notebook 

This notebook implements a **FastAPI-based deployment** for serving the fine-tuned COVID-19 QA model with LoRA adaptation.

## Highlights:

- **Framework & Tools:** FastAPI, Uvicorn, Ngrok (for public access), Nest AsyncIO (to run FastAPI in Colab notebooks).  
- **Model:** LoRA fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` loaded on GPU/CPU.  
- **API Endpoints:**
  - `GET /` – Simple health check (`API is running`).  
  - `POST /qa/` – Accepts a JSON question, returns structured answer.
- **Input/Output Schema:**
  - Input: `{ "question": "Your question here" }`  
  - Output: `{ "answer": "Generated answer here" }`  
- **Pipeline Steps:**
  1. Tokenize input question.
  2. Generate answer using fine-tuned model.
  3. Extract and clean answer text.
  4. Return structured JSON response.
- **Public Access:** Ngrok provides a temporary public URL for testing the API.  
- **Testing:** Example POST request using `curl` demonstrates query-response functionality.  .

