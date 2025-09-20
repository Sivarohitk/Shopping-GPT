import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from src.embeddings import (
    load_image_embeddings,
    load_text_embeddings,
    QueryEmbedder,
    normalize,
)

# ---------- Similarity ----------
def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and all rows in matrix.
    Both must be normalized already (||x||=1).
    Args:
        query_vec (1, D)
        matrix (N, D)
    Returns:
        scores (N,)
    """
    return np.dot(matrix, query_vec.T).squeeze()

# ---------- Recommender Class ----------
class Recommender:
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        # Load precomputed embeddings
        self.img_embs, self.img_idx, self.img_meta = load_image_embeddings(embeddings_dir)
        self.txt_embs, self.txt_idx, self.txt_meta = load_text_embeddings(embeddings_dir)

        # Normalize (safety, though build_embeddings already did)
        self.img_embs = normalize(self.img_embs)
        self.txt_embs = normalize(self.txt_embs)

        # Coerce article_id to int for reliable merges
        for df in (self.img_idx, self.txt_idx):
            if "article_id" in df.columns:
                df["article_id"] = pd.to_numeric(df["article_id"], errors="coerce").astype("Int64")
                df.dropna(subset=["article_id"], inplace=True)
                df["article_id"] = df["article_id"].astype("int64")

        # Shared embedder for user queries
        self.embedder = QueryEmbedder()

    # --- Query by text ---
    def recommend_by_text(
        self, text: str, top_k: int = 10, use_text_embeddings: bool = True
    ) -> pd.DataFrame:
        """Return top_k recommendations given a text query."""
        q_vec = self.embedder.embed_text(text)  # (1, D)

        if use_text_embeddings:
            scores = cosine_similarity(q_vec, self.txt_embs)
            idx = np.argsort(scores)[::-1][:top_k]
            results = self.txt_idx.iloc[idx].copy()
            results["score"] = scores[idx]
        else:
            # Compare text query against image embeddings
            q_vec_imgspace = self.embedder.text_model.encode([text], normalize_embeddings=True)
            scores = cosine_similarity(q_vec_imgspace, self.img_embs)
            idx = np.argsort(scores)[::-1][:top_k]
            results = self.img_idx.iloc[idx].copy()
            results["score"] = scores[idx]

        return results

    # --- Query by image ---
    def recommend_by_image(self, image_path: str, top_k: int = 10) -> pd.DataFrame:
        """Return top_k recommendations given an image query."""
        q_vec = self.embedder.embed_image(image_path)  # (1, D)
        scores = cosine_similarity(q_vec, self.img_embs)
        idx = np.argsort(scores)[::-1][:top_k]

        results = self.img_idx.iloc[idx].copy()
        results["score"] = scores[idx]
        return results

    # --- Hybrid (text + image) ---
    def recommend_hybrid(
        self, text: str, image_path: str, alpha: float = 0.5, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Combine text and image similarity scores by aligning on article_id.
        alpha: weight for text (0-1). (1-alpha) is weight for image.
        """
        # 1) Embed queries
        q_vec_text = self.embedder.embed_text(text)            # (1, D_text)
        q_vec_img  = self.embedder.embed_image(image_path)     # (1, D_img)

        # 2) Scores vs each embedding matrix
        scores_text = cosine_similarity(q_vec_text, self.txt_embs)  # (N_text,)
        scores_img  = cosine_similarity(q_vec_img,  self.img_embs)  # (N_img,)

        # 3) Per-article best score (dedupe)
        df_text = self.txt_idx[["article_id"]].copy()
        df_text["score_text"] = scores_text
        df_text = df_text.groupby("article_id", as_index=False)["score_text"].max()

        df_img = self.img_idx[["article_id"]].copy()
        df_img["score_img"] = scores_img
        df_img = df_img.groupby("article_id", as_index=False)["score_img"].max()

        # 4) Align by article_id; if no overlap, fall back to text-only
        merged = pd.merge(df_text, df_img, on="article_id", how="inner")
        if merged.empty:
            idx = np.argsort(scores_text)[::-1][:top_k]
            out = self.txt_idx.iloc[idx].copy()
            out["score"] = scores_text[idx]
            return out

        # 5) Hybrid score and rank
        merged["score"] = alpha * merged["score_text"] + (1 - alpha) * merged["score_img"]
        merged = merged.sort_values("score", ascending=False).head(top_k)

        # 6) Attach one image_path if available (join_results will also fill later)
        img_paths = self.img_idx[["article_id", "image_path"]].drop_duplicates("article_id")
        merged = merged.merge(img_paths, on="article_id", how="left")

        return merged[["article_id", "image_path", "score"]]

# ---------- Example Usage ----------
if __name__ == "__main__":
    rec = Recommender()

    print("\n=== Text Query: 'red summer dress' ===")
    print(rec.recommend_by_text("red summer dress", top_k=5))

    print("\n=== Image Query: data/processed/images/0129085001.jpg ===")
    print(rec.recommend_by_image("data/processed/images/0129085001.jpg", top_k=5))

    print("\n=== Hybrid Query: text + image ===")
    print(
        rec.recommend_hybrid(
            "summer outfit", "data/processed/images/0129085001.jpg", alpha=0.6, top_k=5
        )
    )
