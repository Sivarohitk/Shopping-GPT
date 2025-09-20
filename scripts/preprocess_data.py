import os
import pandas as pd
import shutil

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess_hm_dataset(
    sample_size=10000,
    copy_images=True
):
    """
    Preprocess the H&M dataset:
    - Load raw CSVs
    - Take a smaller sample of articles
    - Filter customers & transactions to match those articles
    - Optionally copy only sampled images
    - Save reduced files to data/processed/
    """

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if copy_images:
        os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)

    # Paths
    articles_path = os.path.join(RAW_DIR, "articles.csv")
    customers_path = os.path.join(RAW_DIR, "customers.csv")
    transactions_path = os.path.join(RAW_DIR, "transactions_train.csv")

    print("Loading raw CSVs...")
    articles = pd.read_csv(articles_path)
    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)

    print(f"Original Shapes -> Articles: {articles.shape}, Customers: {customers.shape}, Transactions: {transactions.shape}")

    # Sample articles
    sampled_articles = articles.sample(n=sample_size, random_state=42)
    sampled_article_ids = set(sampled_articles["article_id"].unique())

    # Filter transactions to keep only sampled articles
    sampled_transactions = transactions[transactions["article_id"].isin(sampled_article_ids)]

    # Filter customers
    sampled_customer_ids = set(sampled_transactions["customer_id"].unique())
    sampled_customers = customers[customers["customer_id"].isin(sampled_customer_ids)]

    print(f"Sampled Shapes -> Articles: {sampled_articles.shape}, Customers: {sampled_customers.shape}, Transactions: {sampled_transactions.shape}")

    # Save reduced CSVs
    sampled_articles.to_csv(os.path.join(PROCESSED_DIR, "articles_sample.csv"), index=False)
    sampled_customers.to_csv(os.path.join(PROCESSED_DIR, "customers_sample.csv"), index=False)
    sampled_transactions.to_csv(os.path.join(PROCESSED_DIR, "transactions_sample.csv"), index=False)

    # Copy only sampled images (if available and enabled)
    if copy_images:
        image_dir = os.path.join(RAW_DIR, "images")
        target_dir = os.path.join(PROCESSED_DIR, "images")

        print("Copying sampled images... (this may take some time)")
        copied_count = 0

        for article_id in sampled_article_ids:
            # H&M images are named with 7-digit padded article_id + 6-digit color code
            article_str = str(article_id).zfill(10)[:7]  # first 7 digits
            subdir = article_str[:3]  # folder name (first 3 digits)
            image_name = f"{article_str}001.jpg"  # default image naming pattern

            src_path = os.path.join(image_dir, subdir, image_name)
            dest_path = os.path.join(target_dir, image_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                copied_count += 1

        print(f"Copied {copied_count} images to {target_dir}")

    print("Preprocessing complete. Processed files saved in data/processed/")

if __name__ == "__main__":
    preprocess_hm_dataset(sample_size=10000, copy_images=True)
