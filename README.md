# Genia NER Biomedical

This project focuses on building a robust Named Entity Recognition (NER) system for biomedical text using BERT-based models and BIO tagging scheme. It uses the GENIA corpus — a widely used annotated dataset in biomedical NLP — and offers a user-friendly Streamlit interface for real-time entity extraction.

---

## 🧠 Background

In biomedical research and clinical settings, unstructured text like research papers, patient records, and clinical notes contains rich and critical information. Extracting entities such as proteins, DNA, RNA, and cell types helps in:

- **Automated knowledge extraction** from literature
- **Clinical decision support** by identifying symptoms, drugs, or procedures
- **Improving search engines** for medical research (semantic search)

NER is equally vital in **legal**, **finance**, and **cybersecurity** domains for identifying entities like dates, laws, case IDs, names, or sensitive PII.

---

## ❓ Problem Statement

Traditional NER systems trained on general corpora fail to accurately identify domain-specific entities (e.g., "NF-κB", "T-cell") in biomedical texts due to:

- Ambiguous abbreviations and entity overlap
- Lack of domain-specific annotated data
- Poor generalization of generic models in high-stakes fields

---

## ✅ Proposed Solution

We propose a lightweight, trainable, and explainable NER pipeline:

- **Dataset**: GENIA JSON-formatted corpus (`sentence`, `entities`)
- **Model**: BERT-based token classification model trained using HuggingFace `Trainer`
- **Tagging Scheme**: BIO (Beginning, Inside, Outside) format for fine-grained entity boundaries
- **UI**: Streamlit frontend for interactive input and entity extraction
- **Confidence Thresholding**: Allows filtering low-confidence predictions

### Folder Structure:

