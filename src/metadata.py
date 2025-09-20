# src/metadata.py
import os
import pandas as pd

PROCESSED_DIR = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"

# Columns we’ll expose in the UI (use only those that exist)
PREFERRED_COLS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "index_name",
    "colour_group_name",
    "graphical_appearance_name",
    "detail_desc",
]


def _coerce_article_id(df: pd.DataFrame, col: str = "article_id") -> pd.DataFrame:
    """Ensure article_id is int64 for reliable merges."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        # Drop rows where article_id could not be parsed
        df = df[df[col].notna()].copy()
        df[col] = df[col].astype("int64")
    return df


def load_articles(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """
    Load product metadata from data/processed/articles_sample.csv (preferred),
    or fallback to data/processed/articles.csv if present.
    Returns a de-duplicated DataFrame with friendly columns.
    """
    sample_path = os.path.join(processed_dir, "articles_sample.csv")
    full_path = os.path.join(processed_dir, "articles.csv")

    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    elif os.path.exists(full_path):
        df = pd.read_csv(full_path)
    else:
        raise FileNotFoundError(
            f"Could not find {sample_path} (or {full_path}). "
            "Run scripts/preprocess_data.py first."
        )

    df = _coerce_article_id(df, "article_id")

    keep = [c for c in PREFERRED_COLS if c in df.columns]
    if "article_id" not in keep:
        keep = ["article_id"] + keep

    df = df[keep].drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return df


def load_image_index(embeddings_dir: str = EMBEDDINGS_DIR) -> pd.DataFrame:
    """
    Load article_id -> image_path mapping from data/embeddings/image_index.csv.
    Returns empty DataFrame if the file isn't there (e.g., text-only mode).
    """
    idx_path = os.path.join(embeddings_dir, "image_index.csv")
    if not os.path.exists(idx_path):
        return pd.DataFrame(columns=["article_id", "image_path"])

    idx = pd.read_csv(idx_path)
    idx = _coerce_article_id(idx, "article_id")

    # Deduplicate in case of multiple color variants; keep first path
    idx = idx.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return idx[["article_id", "image_path"]]


def attach_image_paths(
    results_df: pd.DataFrame,
    embeddings_dir: str = EMBEDDINGS_DIR,
) -> pd.DataFrame:
    """
    Ensure results have an 'image_path' column by joining with image_index.csv.
    If results already have image_path, they are preserved.
    """
    if results_df is None or results_df.empty:
        return results_df

    results_df = results_df.copy()
    results_df = _coerce_article_id(results_df, "article_id")

    if "image_path" in results_df.columns and results_df["image_path"].notna().any():
        # Already present (e.g., image search); still try to fill any missing
        missing_mask = results_df["image_path"].isna() if "image_path" in results_df else None
    else:
        missing_mask = None

    idx = load_image_index(embeddings_dir)
    if not idx.empty:
        results_df = results_df.merge(idx, on="article_id", how="left", suffixes=("", "_idx"))

        # If there was an existing image_path, prefer it; else use the index path
        if "image_path_x" in results_df.columns and "image_path_y" in results_df.columns:
            results_df["image_path"] = results_df["image_path_x"].fillna(results_df["image_path_y"])
            results_df = results_df.drop(columns=["image_path_x", "image_path_y"])

    return results_df


def join_results(
    results_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    attach_images: bool = True,
    embeddings_dir: str = EMBEDDINGS_DIR,
) -> pd.DataFrame:
    """
    Attach human-friendly product fields (name/type/color/etc.) to the recommender results.
    Optionally also attach image paths from the embeddings index.
    """
    if results_df is None or results_df.empty:
        return results_df

    results_df = _coerce_article_id(results_df.copy(), "article_id")
    articles_df = _coerce_article_id(articles_df.copy(), "article_id")

    out = results_df.merge(articles_df, on="article_id", how="left")

    if attach_images:
        out = attach_image_paths(out, embeddings_dir=embeddings_dir)

    # Reorder columns for nicer display
    ordered = [
        c for c in [
            "article_id",
            "score",
            "image_path",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "index_name",
            "colour_group_name",
            "graphical_appearance_name",
            "detail_desc",
        ] if c in out.columns
    ]
    other = [c for c in out.columns if c not in ordered]
    return out[ordered + other].reset_index(drop=True)
