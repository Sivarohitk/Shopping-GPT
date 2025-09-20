# pages/about.py
import os
import numpy as np
import pandas as pd
import streamlit as st

st.title("ℹ️ About • StyleFinder AI")
st.caption("Multimodal fashion search — NLP + Computer Vision + a grounded LLM assistant.")

# --- Helpers to load optional artifacts safely ---
PROC = "data/processed"
EMB = "data/embeddings"

def safe_read_csv(path: str):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def image_count(dirpath: str) -> int:
    if not os.path.isdir(dirpath):
        return 0
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return sum(1 for f in os.listdir(dirpath) if f.lower().endswith(exts))

def np_shape(path: str):
    if os.path.exists(path):
        try:
            arr = np.load(path)
            return tuple(arr.shape)
        except Exception:
            return None
    return None

# --- Load optional sample stats ---
articles = safe_read_csv(os.path.join(PROC, "articles_sample.csv"))
customers = safe_read_csv(os.path.join(PROC, "customers_sample.csv"))
transactions = safe_read_csv(os.path.join(PROC, "transactions_sample.csv"))

img_dir = os.path.join(PROC, "images")
n_images = image_count(img_dir)

img_emb_shape = np_shape(os.path.join(EMB, "image_embeddings.npy"))
txt_emb_shape = np_shape(os.path.join(EMB, "text_embeddings.npy"))

# --- Overview ---
st.subheader("What is StyleFinder AI?")
st.write("""
**StyleFinder AI** is a full-stack, open-source fashion recommendation app:
- **Text search (NLP)** — find items with sentence embeddings (Sentence Transformers).
- **Image search (CV)** — find visually similar items with CLIP image embeddings.
- **Hybrid search** — blend text + image with a simple weight slider.
- **Grounded LLM assistant** — a small local model answers questions using the retrieved items.
- **Cloud-deployable** — runs on Streamlit Cloud or Hugging Face Spaces (CPU, free tier).
""")

# --- Quick stats (if files exist) ---
st.subheader("Project snapshot (from your processed sample)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles (sample)", f"{len(articles):,}" if articles is not None else "—")
c2.metric("Customers (sample)", f"{len(customers):,}" if customers is not None else "—")
c3.metric("Transactions (sample)", f"{len(transactions):,}" if transactions is not None else "—")
c4.metric("Sample images", f"{n_images:,}" if n_images else "—")

c5, c6 = st.columns(2)
c5.metric("Image embeddings", f"{img_emb_shape}" if img_emb_shape else "—")
c6.metric("Text embeddings", f"{txt_emb_shape}" if txt_emb_shape else "—")

# --- How it works ---
st.subheader("How it works (end-to-end)")
st.markdown("""
1. **Preprocess**: From the H&M dataset, create a **small sample** of articles + matching transactions and **copy only those images** to `data/processed/`.
2. **Embed**: Build **image embeddings** with CLIP and **text embeddings** with Sentence Transformers; save to `data/embeddings/`.
3. **Search**: For a user query (text or image), compute a query embedding and run **cosine similarity** against the precomputed vectors to get top-K items.
4. **Enrich**: Join results with product metadata (name, type, color, description) for readable cards.
5. **Assist**: Pass the retrieved items to a **small LLM** to generate grounded styling suggestions.
6. **UI**: Streamlit renders an interactive gallery (text, image, or hybrid search) + a chat box.
""")

# --- Run locally ---
st.subheader("Run locally")
st.code(
    "python scripts/preprocess_data.py\n"
    "python scripts/build_embeddings.py\n"
    "streamlit run app.py",
    language="bash"
)

# --- Deploy ---
st.subheader("Deploy (free)")
st.markdown("""
- **Streamlit Community Cloud**: push your repo with `requirements.txt` and `app.py` → deploy from GitHub.  
- **Hugging Face Spaces**: create a Space (Streamlit), add your repo → automatic build on CPU (16 GB RAM).
""")

# --- Tech stack ---
st.subheader("Tech stack")
st.markdown("""
- **Python**, **NumPy**, **Pandas**, **tqdm**  
- **PyTorch**, **Transformers**, **Sentence-Transformers**, **CLIP**  
- **Streamlit** (frontend), optional **FAISS** (faster vector search)  
- Optional **Docker** for containerized deployment
""")

# --- Notes ---
st.subheader("Notes")
st.markdown("""
- Uses **open-source models only** (no paid APIs).
- Keep the **full dataset** in `data/raw/` (not in Git). Work with the **sample** in `data/processed/` for development and cloud demos.
- Embeddings are normalized for cosine similarity; results are reproducible given the same sample.
""")
