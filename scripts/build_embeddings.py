import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image

# CLIP (vision + text tokenizer)
from transformers import CLIPProcessor, CLIPModel

# Sentence-Transformers for compact text embeddings
from sentence_transformers import SentenceTransformer


# ----------- Paths -----------
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
IMAGES_SUBDIR = "images"  # images live under data/processed/images/


# ----------- Helpers -----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_image_filename_from_article(article_id: int) -> str:
    """
    H&M images are stored as <first 7 digits><color> .jpg.
    Example: article_id 129085 (or 0000129085) -> "0129085" + "001" = "0129085001.jpg"
    In our preprocess step we copied default '...001.jpg' when available.
    """
    art7 = str(article_id).zfill(10)[:7]
    return f"{art7}001.jpg"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_articles_df(processed_dir: str) -> pd.DataFrame:
    # Prefer the sampled CSV; fall back to full if user renamed
    sample_path = os.path.join(processed_dir, "articles_sample.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    # Fallback names (not recommended)
    alt1 = os.path.join(processed_dir, "articles.csv")
    if os.path.exists(alt1):
        return pd.read_csv(alt1)
    raise FileNotFoundError(
        f"Could not find articles_sample.csv in {processed_dir}. "
        "Run scripts/preprocess_data.py first."
    )


def compose_article_text(row: pd.Series) -> str:
    """
    Build a concise text string per article to embed:
    prefer detail_desc if present, otherwise combine metadata columns.
    """
    fields = []
    for col in [
        "detail_desc",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "index_name",
        "colour_group_name",
        "graphical_appearance_name",
    ]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            fields.append(str(row[col]))
    if not fields:
        return f"article {row.get('article_id')}"
    return " | ".join(fields)


# ----------- Image Embeddings with CLIP -----------
def build_image_embeddings(
    df: pd.DataFrame,
    images_dir: str,
    out_dir: str,
    batch_size: int = 64,
    model_name: str = "openai/clip-vit-base-patch32",
):
    device = choose_device()
    print(f"[Image] Using device: {device}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    article_ids = []
    image_paths = []
    # Resolve paths and keep only existing images
    for _, row in df.iterrows():
        art_id = int(row["article_id"])
        filename = build_image_filename_from_article(art_id)
        path = os.path.join(images_dir, filename)
        if os.path.exists(path):
            article_ids.append(art_id)
            image_paths.append(path)
        # else: silently skip missing image

    if not image_paths:
        raise RuntimeError(
            f"No images found under {images_dir}. "
            "Ensure preprocess_data.py copied sampled images."
        )

    print(f"[Image] Found {len(image_paths)} images to embed")

    all_embeds = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="[Image] Batching"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                # Skip unreadable images
                images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize to unit length (common for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeds.append(image_features.cpu().numpy())

    image_embeddings = np.concatenate(all_embeds, axis=0)

    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "image_embeddings.npy"), image_embeddings)

    # Save index mapping
    idx_df = pd.DataFrame(
        {"article_id": np.array(article_ids, dtype=np.int64), "image_path": image_paths}
    )
    idx_df.to_csv(os.path.join(out_dir, "image_index.csv"), index=False)

    meta = {
        "model": model_name,
        "dim": int(image_embeddings.shape[1]),
        "count": int(image_embeddings.shape[0]),
        "device": str(device),
        "normalized": True,
    }
    with open(os.path.join(out_dir, "image_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[Image] Saved embeddings: {image_embeddings.shape} to {out_dir}/image_embeddings.npy"
    )


# ----------- Text Embeddings with Sentence-Transformers -----------
def build_text_embeddings(
    df: pd.DataFrame,
    out_dir: str,
    batch_size: int = 256,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Text] Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    # Compose text per article
    texts = df.apply(compose_article_text, axis=1).tolist()
    article_ids = df["article_id"].astype("int64").tolist()

    print(f"[Text] Encoding {len(texts)} texts...")
    # sentence-transformers handles batching internally via encode()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # already L2 normalize for cosine similarity
    )

    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "text_embeddings.npy"), embeddings)

    # Save index mapping
    idx_df = pd.DataFrame({"article_id": article_ids, "text": texts})
    idx_df.to_csv(os.path.join(out_dir, "text_index.csv"), index=False)

    meta = {
        "model": model_name,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "normalized": True,
    }
    with open(os.path.join(out_dir, "text_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[Text] Saved embeddings: {embeddings.shape} to {out_dir}/text_embeddings.npy"
    )


# ----------- Main -----------
def main():
    parser = argparse.ArgumentParser(description="Build image & text embeddings")
    parser.add_argument(
        "--processed_dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Directory with processed CSVs and images/",
    )
    parser.add_argument(
        "--embeddings_dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Output directory for .npy and index files",
    )
    parser.add_argument(
        "--image_model",
        default="openai/clip-vit-base-patch32",
        help="HF model id for CLIP vision encoder",
    )
    parser.add_argument(
        "--text_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model id",
    )
    parser.add_argument(
        "--image_batch_size", type=int, default=64, help="Batch size for CLIP"
    )
    parser.add_argument(
        "--text_batch_size", type=int, default=256, help="Batch size for text encoder"
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip building image embeddings (text only)",
    )
    parser.add_argument(
        "--skip_text",
        action="store_true",
        help="Skip building text embeddings (images only)",
    )

    args = parser.parse_args()

    ensure_dir(args.embeddings_dir)

    df = load_articles_df(args.processed_dir)

    if not args.skip_images:
        images_dir = os.path.join(args.processed_dir, IMAGES_SUBDIR)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Images directory not found at {images_dir}. "
                "Run preprocess_data.py with copy_images=True."
            )
        build_image_embeddings(
            df=df,
            images_dir=images_dir,
            out_dir=args.embeddings_dir,
            batch_size=args.image_batch_size,
            model_name=args.image_model,
        )

    if not args.skip_text:
        build_text_embeddings(
            df=df,
            out_dir=args.embeddings_dir,
            batch_size=args.text_batch_size,
            model_name=args.text_model,
        )

    print("\n✅ Done. Embeddings saved in:", args.embeddings_dir)


if __name__ == "__main__":
    main()
