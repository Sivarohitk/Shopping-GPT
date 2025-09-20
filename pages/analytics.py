# pages/analytics.py
import os
import pandas as pd
import numpy as np
import streamlit as st

PROC_DIR = "data/processed"
ARTICLES = os.path.join(PROC_DIR, "articles_sample.csv")
CUSTOMERS = os.path.join(PROC_DIR, "customers_sample.csv")
TRANSACTIONS = os.path.join(PROC_DIR, "transactions_sample.csv")

st.set_page_config(page_title="Analytics • StyleFinder AI", page_icon="📊", layout="wide")
st.title("📊 Dataset Analytics (Sample)")

# --------------------------- helpers ---------------------------

@st.cache_data(show_spinner=False)
def safe_read_csv(path: str, usecols=None, parse_dates=None):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_data():
    # Load what we need with graceful fallbacks
    articles = safe_read_csv(
        ARTICLES,
        usecols=None  # keep all, we’ll select later
    )
    customers = safe_read_csv(
        CUSTOMERS,
        usecols=None
    )
    # For transactions, load only likely-needed columns if present
    trx_cols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    transactions = safe_read_csv(
        TRANSACTIONS,
        usecols=None,
        parse_dates=["t_dat"]
    )
    # Coerce dtypes
    if articles is not None and "article_id" in articles.columns:
        articles["article_id"] = pd.to_numeric(articles["article_id"], errors="coerce").astype("Int64")
        articles = articles[articles["article_id"].notna()].copy()
        articles["article_id"] = articles["article_id"].astype("int64")
    if transactions is not None and "article_id" in transactions.columns:
        transactions["article_id"] = pd.to_numeric(transactions["article_id"], errors="coerce").astype("Int64")
        transactions = transactions[transactions["article_id"].notna()].copy()
        transactions["article_id"] = transactions["article_id"].astype("int64")
    return articles, customers, transactions

def has_cols(df: pd.DataFrame, cols):
    return df is not None and all(c in df.columns for c in cols)

# --------------------------- load ---------------------------

articles, customers, transactions = load_data()

if articles is None and transactions is None:
    st.warning("Could not find processed CSVs. Run `scripts/preprocess_data.py` first.")
    st.stop()

# --------------------------- high-level snapshot ---------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles (sample)", f"{len(articles):,}" if articles is not None else "—")
c2.metric("Customers (sample)", f"{len(customers):,}" if customers is not None else "—")
c3.metric("Transactions (sample)", f"{len(transactions):,}" if transactions is not None else "—")
n_types = articles["product_type_name"].nunique() if has_cols(articles, ["product_type_name"]) else 0
c4.metric("Unique product types", f"{n_types:,}" if n_types else "—")

st.markdown("---")

# --------------------------- filters ---------------------------

with st.expander("Filters"):
    # Date filter (if transactions present)
    if has_cols(transactions, ["t_dat"]):
        min_d, max_d = transactions["t_dat"].min(), transactions["t_dat"].max()
        start_d, end_d = st.slider(
            "Transaction date range",
            min_value=min_d.to_pydatetime() if pd.notna(min_d) else None,
            max_value=max_d.to_pydatetime() if pd.notna(max_d) else None,
            value=(
                min_d.to_pydatetime() if pd.notna(min_d) else None,
                max_d.to_pydatetime() if pd.notna(max_d) else None,
            ),
            disabled=(min_d is pd.NaT or max_d is pd.NaT)
        )
        if start_d and end_d:
            mask = (transactions["t_dat"] >= pd.to_datetime(start_d)) & (transactions["t_dat"] <= pd.to_datetime(end_d))
            trx_f = transactions.loc[mask].copy()
        else:
            trx_f = transactions.copy()
    else:
        trx_f = transactions

    # Optional sample size limit for speed
    max_rows = st.number_input("Limit rows for faster charts (0 = no limit)", min_value=0, value=0, step=1000)
    if max_rows and trx_f is not None and len(trx_f) > max_rows:
        trx_f = trx_f.sample(n=max_rows, random_state=42)

# --------------------------- charts: articles ---------------------------

st.subheader("Articles overview")
cc1, cc2 = st.columns(2)

with cc1:
    st.markdown("**Top product types**")
    if has_cols(articles, ["product_type_name"]):
        top_types = (
            articles["product_type_name"]
            .value_counts()
            .head(15)
            .rename_axis("product_type_name")
            .reset_index(name="count")
        )
        st.bar_chart(top_types.set_index("product_type_name"))
    else:
        st.info("Column `product_type_name` not found in articles_sample.csv")

with cc2:
    st.markdown("**Color distribution**")
    if has_cols(articles, ["colour_group_name"]):
        colors = (
            articles["colour_group_name"]
            .value_counts()
            .head(15)
            .rename_axis("colour_group_name")
            .reset_index(name="count")
        )
        st.bar_chart(colors.set_index("colour_group_name"))
    else:
        st.info("Column `colour_group_name` not found in articles_sample.csv")

st.markdown("---")

# --------------------------- charts: transactions ---------------------------

st.subheader("Transactions overview")

tc1, tc2 = st.columns(2)

with tc1:
    st.markdown("**Most purchased articles (by count)**")
    if has_cols(trx_f, ["article_id"]):
        top_items = (
            trx_f["article_id"]
            .value_counts()
            .head(20)
            .rename_axis("article_id")
            .reset_index(name="purchases")
        )
        st.dataframe(top_items, use_container_width=True, hide_index=True)
    else:
        st.info("Transactions file missing `article_id`.")

with tc2:
    st.markdown("**Sales channel mix**")
    if has_cols(trx_f, ["sales_channel_id"]):
        channel_mix = (
            trx_f["sales_channel_id"]
            .value_counts()
            .rename_axis("sales_channel_id")
            .reset_index(name="count")
        )
        st.bar_chart(channel_mix.set_index("sales_channel_id"))
    else:
        st.info("Transactions file missing `sales_channel_id`.")

tc3, tc4 = st.columns(2)

with tc3:
    st.markdown("**Purchases over time**")
    if has_cols(trx_f, ["t_dat"]):
        # Daily or weekly aggregation depending on date span
        df_time = trx_f.copy()
        df_time["t_dat"] = pd.to_datetime(df_time["t_dat"], errors="coerce")
        if df_time["t_dat"].notna().sum() > 0:
            # Resample by week for smoother chart
            ts = (
                df_time.set_index("t_dat")
                .assign(cnt=1)["cnt"]
                .resample("W")
                .sum()
                .rename("purchases")
            )
            st.line_chart(ts)
        else:
            st.info("No valid dates in `t_dat` to chart.")
    else:
        st.info("Transactions file missing `t_dat`.")

with tc4:
    st.markdown("**Top product groups (by transactions)**")
    if has_cols(trx_f, ["article_id"]) and has_cols(articles, ["article_id", "product_group_name"]):
        merged = trx_f.merge(articles[["article_id", "product_group_name"]], on="article_id", how="left")
        grp = (
            merged["product_group_name"]
            .fillna("Unknown")
            .value_counts()
            .head(12)
            .rename_axis("product_group_name")
            .reset_index(name="count")
        )
        st.bar_chart(grp.set_index("product_group_name"))
    else:
        st.info("Needed columns not found to compute product group counts.")

st.markdown("---")

# --------------------------- detail table ---------------------------

st.subheader("Browse articles (sample)")
if articles is not None:
    show_cols = [c for c in [
        "article_id", "prod_name", "product_type_name", "product_group_name",
        "index_name", "colour_group_name", "graphical_appearance_name"
    ] if c in articles.columns]
    st.dataframe(articles[show_cols].head(500), use_container_width=True, hide_index=True)
else:
    st.info("Articles file not found.")
