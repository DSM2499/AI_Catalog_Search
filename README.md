# 🛍️ AI-Powered Semantic Search for Fashion Products

## 🚀 How to Run the Project
1. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**
   - Ensure `styles.csv` is placed under `/data`
   - Run the data cleaning script:
     ```bash
     python utils/data_loader.py
     ```

4. **Generate Embeddings**
   ```bash
   python search/semantic_search.py
   ```

5. **Build FAISS Index**
   ```bash
   python search/semantic_indexer.py
   ```

6. **Launch Streamlit App**
   ```bash
   streamlit run app/main_app.py
   ```

---

## 📚 Project Overview

This project enables intelligent product discovery using semantic understanding of user queries. It combines sentence embeddings with FAISS approximate nearest neighbor indexing to deliver fast and meaningful search results.

The app allows users to search for items like _“red sneakers under $50 for men”_ and returns top product matches with associated images, metadata, and similarity scores.

---

## 🧠 Tech Stack and Theoretical Foundations

| Category       | Tools / Concepts                                              |
|----------------|---------------------------------------------------------------|
| **Languages**  | Python 3.10                                                   |
| **Libraries**  | `streamlit`, `pandas`, `numpy`, `faiss`, `sentence-transformers`, `PIL`, `scikit-learn` |
| **NLP Models** | `all-MiniLM-L6-v2` from HuggingFace Sentence Transformers     |
| **Vector Search** | FAISS (Facebook AI Similarity Search)                    |
| **Interface**  | Streamlit with dynamic image rendering and filtering          |
| **Search Theory** | Semantic similarity, cosine distance, ANN indexing       |


---

## 🖼️ Screenshots

> Insert your screenshots below these headings once the app is running

### 🎯 Search Results with Images
![Search Results](https://github.com/DSM2499/AI_Catalog_Search/blob/main/Screenshots/search_query.png)

### 🧾 Sidebar Filters
![Filters](https://github.com/DSM2499/AI_Catalog_Search/blob/main/Screenshots/filters.png)

---
## 💡 Use Cases

- Fashion e-commerce sites wanting intelligent search capabilities
- Virtual assistants for product recommendations
- Research on zero-shot product classification or entity linking
- A/B testing semantic vs lexical search performance
- Entry point for integrating with larger recommender systems

---

## 🏁 Conclusion

This project demonstrates a complete end-to-end AI system — from data preparation and semantic embedding to scalable vector search and interactive UI. It not only shows off NLP and vector search expertise but also highlights practical UI integration, performance awareness (M2 Mac compatible), and production-readiness.

Link to the datset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---
