import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer


# Default paths
EMBEDDINGS_DIR = "data/embeddings"


# ---------- Utilities ----------
def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 normalize vectors row-wise."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


# ---------- Load Precomputed Embeddings ----------
def load_image_embeddings(embeddings_dir: str = EMBEDDINGS_DIR):
    """
    Load precomputed image embeddings and index.
    Returns:
        embeddings (np.ndarray), index_df (pd.DataFrame), meta (dict)
    """
    emb_path = os.path.join(embeddings_dir, "image_embeddings.npy")
    idx_path = os.path.join(embeddings_dir, "image_index.csv")
    meta_path = os.path.join(embeddings_dir, "image_meta.json")

    embeddings = np.load(emb_path)
    index_df = pd.read_csv(idx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return embeddings, index_df, meta


def load_text_embeddings(embeddings_dir: str = EMBEDDINGS_DIR):
    """
    Load precomputed text embeddings and index.
    Returns:
        embeddings (np.ndarray), index_df (pd.DataFrame), meta (dict)
    """
    emb_path = os.path.join(embeddings_dir, "text_embeddings.npy")
    idx_path = os.path.join(embeddings_dir, "text_index.csv")
    meta_path = os.path.join(embeddings_dir, "text_meta.json")

    embeddings = np.load(emb_path)
    index_df = pd.read_csv(idx_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return embeddings, index_df, meta


# ---------- Embed New Queries ----------
class QueryEmbedder:
    """Helper for embedding new text or images with the same models."""

    def __init__(
        self,
        image_model_name="openai/clip-vit-base-patch32",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Load models
        self.clip_model = CLIPModel.from_pretrained(image_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.text_model = SentenceTransformer(text_model_name, device=device)

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.text_model.encode([text], normalize_embeddings=True)
        return emb  # shape (1, D)

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()  # shape (1, D)


# ---------- Example Usage ----------
if __name__ == "__main__":
    # Load precomputed embeddings
    img_embs, img_idx, img_meta = load_image_embeddings()
    txt_embs, txt_idx, txt_meta = load_text_embeddings()

    print(f"Loaded {img_embs.shape[0]} image embeddings of dim {img_embs.shape[1]}")
    print(f"Loaded {txt_embs.shape[0]} text embeddings of dim {txt_embs.shape[1]}")

    # Test query embedding
    embedder = QueryEmbedder()
    q1 = embedder.embed_text("red summer dress")
    q2 = embedder.embed_image("data/processed/images/0129085001.jpg")

    print("Query embeddings ready:", q1.shape, q2.shape)
