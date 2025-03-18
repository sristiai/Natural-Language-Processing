
# Project Name
Here is a README file for your project:  

---

# **Medical RAG System**  
**Author:** Sristi Bhadani  
**Supervisor:** Prof. Dr. Patrick Levi  
**Institution:** Ostbayerische Technische Hochschule Amberg-Weiden  
**Department:** Electrical Engineering, Media, and Computer Science  
**Date:** July 3, 2024  

## **Project Overview**  
This project focuses on the design, development, and optimization of a **Retrieval-Augmented Generation (RAG) system** for efficiently processing large-scale biomedical datasets. The system aims to retrieve **contextually relevant information** for multiple-choice medical questions (MCQs) to assist medical professionals such as physicians, nurses, and researchers.  

## **Objectives**  
- **Efficient Context Retrieval**: Improve accuracy and relevance in retrieving medical knowledge.  
- **Scalability and Performance**: Optimize computational efficiency for large biomedical datasets.  
- **Advanced NLP Techniques**: Utilize embedding generation, similarity search, and large language model (LLM) integration.  

## **Methodology**  
The system was developed using a structured multi-stage approach:  

1. **Data Processing**  
   - Extracted and cleaned biomedical datasets from Hugging Face and JSON files.  
   - Standardized question-answer formatting.  

2. **Initial Retrieval Model**  
   - **Chunking**: Divided data into question-answer pairs.  
   - **Embedding Creation**: Used `all-MiniLM-L6-v2` model for generating embeddings.  
   - **Similarity Search**: Employed FAISS for retrieving the most relevant chunks.  
   - **Response Generation**: Integrated LLM (`google/gemma-2b-it`) for answering queries.  
   - **Token Trimming**: Managed input length constraints for LLM processing.  

3. **Enhanced RAG Techniques**  
   - **Re-ranking**: Used `bert-base-uncased` model to refine retrieved chunks.  
   - **Autocut**: Applied threshold-based filtering using `cross-encoder/ms-marco-MiniLM-L-6-v2`.  
   - **Summarization**: Employed `T5-small` model for concise output generation.  

4. **Modified Embedding Strategy**  
   - **Multi-Level Filtering**: Sequential keyword-based filtering for improved precision.  
   - **Domain-Specific Embeddings**: Implemented `MedEmbed-small-v0.1` for enhanced medical data representation.  
   - **Compressor Integration**: Used `FlashrankRerank` to refine retrieved documents dynamically.  

## **Results & Analysis**  
The system demonstrated:  
✅ **Higher retrieval accuracy** with **multi-level filtering and domain-specific embeddings**.  
✅ **Reduced noise and irrelevant context** using **reranking and summarization**.  
✅ **Efficient computational performance** due to optimized embedding techniques.  

However, certain challenges remain:  
⚠️ **Handling rare genetic conditions** due to limited dataset representation.  
⚠️ **Interpreting complex microbiological and pathology data** for specific diagnoses.  

## **Future Improvements**  
- **Topic Modeling**: Cluster datasets into meaningful topic-based groups for improved retrieval.  
- **Advanced Chunking**: Experiment with alternative strategies to balance precision and recall.  
- **Scalability**: Optimize framework for real-time clinical applications.  

## **Usage**  
### **Prerequisites**  
- Python 3.8+  
- Required libraries: `Hugging Face Transformers`, `FAISS`, `LangChain`, `PyTorch`  

### **Setup**  
```bash
pip install transformers faiss-cpu langchain torch
```
  
### **Run the System**  
```python
python main.py --query "What is the best treatment for ischemic stroke?"
```

## **References**  
- **Hugging Face Models**: `google/gemma-2b-it`, `MedEmbed-small-v0.1`  
- **Papers Referenced**: Advanced RAG Techniques, Dense and Sparse Embeddings, etc.  


## 📊 Datasets
The following publicly available datasets were used for training and evaluation:
1. [**MedQA-USMLE Dataset**](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) : Multiple-choice questions for medical reasoning and knowledge.
2. [**Medical Meadow Dataset**](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc) : Contextual biomedical data for retrieval tasks.
3. [**PubMedQA Dataset**](https://pubmed.ncbi.nlm.nih.gov/) : Clinical questions and answers sourced from PubMed articles.


## Installation
1. Create virtual environment
    pip install virtualenv
    python3.11 -m venv env_name
    # Activating virtual Environment
    source env_name/bin/activate
2. Install the dependencies using 
    pip -r requirements.txt

## Runnning the file 
    # Change the path names to run it in the files.
    python filename.py




