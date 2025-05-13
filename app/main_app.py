import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from search.semantic_indexer import FaissSemanticSearch
import faiss
from streamlit_image_zoom import image_zoom
from utils.render_product_card import render_product_card
#Load Resources
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_metadata():
    df = pd.read_csv("embeddings/product_metadata.csv")
    return df

@st.cache_resource
def load_index(dim):
    searcher = FaissSemanticSearch(dim = dim)
    searcher.load("embeddings/faiss.index", "embeddings/faiss_ids.npy")
    return searcher

#--------------------------------
#App config
st.set_page_config(page_title="AI-Powered Semantic Product Search", layout="wide")
st.title("🛍️ Semantic Search Engine for Fashion Products")

#Load Model, metadata, and Index
model = load_model()
metadata = load_metadata()
index = load_index(dim = 384) #Hardcoded for now and to ensure consistency

#---------------------------
#Sidebar filters

st.sidebar.header("🔍 Filters")
gender = ["All"] + sorted(metadata['gender'].dropna().unique().tolist())
categories = ["All"] + sorted(metadata['category'].dropna().unique().tolist())
colors = ["All"] + sorted(metadata['color'].dropna().unique().tolist())

selected_gender = st.sidebar.selectbox("Gender", gender)
selected_category = st.sidebar.selectbox("Category", categories)
selected_color = st.sidebar.selectbox("Color", colors)

#--------------------------------
#Query Input

query = st.text_input("Enter a product query (e.g., 'red sneakers for men')", "")
if query:
    query_embedding = model.encode(query, convert_to_numpy = True)
    ids, scores = index.query(query_embedding, top_k = 20)

    results = metadata.iloc[ids].copy()
    results['score'] = scores

    # Apply sidebar filters
    if selected_gender != "All":
        results = results[results["gender"] == selected_gender.lower()]
    if selected_category != "All":
        results = results[results["category"] == selected_category.lower()]
    if selected_color != "All":
        results = results[results["color"] == selected_color.lower()]
    
    st.subheader(f"Showing {len(results)} results for: _{query}_")
    
    cols = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        col = cols[i % 2]
        with col:
            render_product_card(row)

            #Metadata
            st.markdown(
                f"**Category:** {row['category'].capitalize()}  \n"
                f"**Color:** {row['color'].capitalize()}  \n"
                f"**Gender:** {row['gender'].capitalize()}  \n"
                f"**Usage:** {row['usage'].capitalize()}"
            )

#--------------------------------
#Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, FAISS, and Sentence Transformers")

