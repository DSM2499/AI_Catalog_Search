import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_csv = "data/clean_products.csv",
                        output_npy = "embeddings/product_vectors.npy",
                        output_meta = "embeddings/product_metadata.csv",
                        model_name = "all-MiniLM-L6-v2"):
    #Load cleaned data
    df = pd.read_csv(input_csv)
    
    titles = df['title'].tolist()

    #Load Model
    print(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    #Generate embeddings
    print("[INFO] Generating embeddings...")
    embeddings = model.encode(titles, show_progress_bar = True, convert_to_numpy = True, batch_size = 64)

    #Save embeddings and metadata
    os.makedirs("embeddings", exist_ok = True)
    np.save(output_npy, embeddings)
    df.to_csv(output_meta, index = False)

    print(f"[INFO] Embeddings saved to: {output_npy}")
    print(f"[INFO] Metadata saved to: {output_meta}")


if __name__ == "__main__":
    generate_embeddings()
    