
import faiss
import numpy as np
import os
import pandas as pd

class FaissSemanticSearch:
    def __init__(self, dim = 384):
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = None
        self.dim = dim

    def build_index(self, embeddings, ids = None):
        print("[INFO] Building FAISS index...")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.id_map = np.array(ids if ids is not None else range(len(embeddings)))
        print(f"[INFO] Index built with {len(embeddings)} vectors")

    def save(self, index_path = 'embeddings/faiss.index', ids_path = 'embeddings/faiss_ids.npy'):
        faiss.write_index(self.index, index_path)
        np.save(ids_path, self.id_map)
        print(f"[INFO] Index saved to: {index_path}")
        print(f"[INFO] IDs saved to: {ids_path}")

    def load(self, index_path="embeddings/faiss.index", ids_path="embeddings/faiss_ids.npy"):
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(ids_path)
        print("[INFO] FAISS index loaded.")
    
    def query(self, query_embedding, top_k = 5):
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(query_embedding, top_k)
        return self.id_map[I[0]], D[0]

if __name__ == "__main__":
    embeddings = np.load("embeddings/product_vectors.npy")
    metadata = pd.read_csv("embeddings/product_metadata.csv")

    search_index = FaissSemanticSearch(dim=embeddings.shape[1])
    search_index.build_index(embeddings)
    search_index.save()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "red sneakers for men"
    query_embedding = model.encode(query, convert_to_numpy = True)

    ids, scores = search_index.query(query_embedding, top_k = 5)
    print("\nTop 5 matches:")
    print(metadata.iloc[ids][['title', 'category', 'color']])