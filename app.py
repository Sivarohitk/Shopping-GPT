# app.py
import os
import sys
import tempfile
import pandas as pd
import streamlit as st

# ---- Robust imports (works whether 'src' is a package or a folder) ----
try:
    from src.recommender import Recommender
    from src.metadata import load_articles, join_results
    from src.chatbot import FashionAssistant
except Exception:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from recommender import Recommender
    from metadata import load_articles, join_results
    from chatbot import FashionAssistant


# ------------------------ Page config & styles ------------------------
st.set_page_config(page_title="Shopping GPT", page_icon="🛍️", layout="wide")

CARD_CSS = """
<style>
.chat-row { display: flex; gap: 8px; align-items: center; }
.attach-chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border:1px solid #e5e5e5; border-radius:14px;
  background:#fafafa; font-size:0.9rem;
}
.result-card { border:1px solid #e6e6e6; border-radius:12px; padding:8px; margin-bottom:12px; background:white; }
.result-name { font-weight:600; margin-top:6px; }
.result-meta { color:#555; font-size:0.9rem; }
.score-pill { display:inline-block; background:#f5f5f5; padding:2px 8px; border-radius:12px; font-size:0.8rem; margin-top:4px; }
.plus-btn { font-size:20px; padding:4px 10px; border-radius:10px; border:1px solid #e5e5e5; background:#fff; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


# ------------------------ Cached resources ------------------------
@st.cache_resource
def load_recommender():
    return Recommender(embeddings_dir="data/embeddings")

@st.cache_resource
def load_articles_df():
    return load_articles(processed_dir="data/processed")

@st.cache_resource
def load_assistant():
    # If you updated chatbot.py to TinyLlama default, no arg needed.
    # Otherwise, set model_id here.
    return FashionAssistant()

rec = load_recommender()
articles_df = load_articles_df()
assistant = load_assistant()


# ------------------------ Session state ------------------------
if "messages" not in st.session_state:
    # [{role, text, image_path, results_records}]
    st.session_state.messages = []
if "attach_image_tmp" not in st.session_state:
    st.session_state.attach_image_tmp = None
# key to force-reset the uploader widget on "Remove"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K results", 5, 30, 12, 1)
    alpha = st.slider("Hybrid: weight for text", 0.0, 1.0, 0.5, 0.05)
    show_detail = st.checkbox("Show detailed description", value=False)
    st.markdown("---")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.attach_image_tmp = None
        st.session_state.uploader_key += 1
        st.rerun()


# ------------------------ Helpers ------------------------
def render_results_grid(df: pd.DataFrame, show_detail: bool = False):
    if df is None or df.empty:
        return
    cols = st.columns(5)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 5]:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            img_path = row.get("image_path")
            if isinstance(img_path, str) and os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.image("https://via.placeholder.com/300x380.png?text=No+Image", use_column_width=True)
            name = row.get("prod_name", "Unknown")
            ptype = row.get("product_type_name", "")
            color = row.get("colour_group_name", "")
            st.markdown(f'<div class="result-name">{name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-meta">{ptype} • {color}</div>', unsafe_allow_html=True)
            if "score" in row:
                st.markdown(
                    f'<div class="score-pill">ID: {row["article_id"]} • Score: {row["score"]:.3f}</div>',
                    unsafe_allow_html=True,
                )
            if show_detail:
                desc = row.get("detail_desc", "")
                if isinstance(desc, str) and desc.strip():
                    with st.expander("Details"):
                        st.write(desc)
            st.markdown("</div>", unsafe_allow_html=True)

def df_to_records_for_store(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    keep = [
        "article_id", "score", "image_path",
        "prod_name", "product_type_name", "colour_group_name", "detail_desc"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].to_dict(orient="records")

def records_to_df(records):  # -> DataFrame
    return pd.DataFrame.from_records(records) if records else pd.DataFrame()

def infer_alpha(user_text: str | None, slider_alpha: float) -> float:
    """If the text is generic ('this', 'like this'), make image dominate."""
    if not user_text:
        return slider_alpha
    t = user_text.strip().lower()
    generic = ["this", "like this", "same as this", "similar to this", "like that"]
    if any(p == t or p in t for p in generic):
        return min(slider_alpha, 0.20)
    if len(t.split()) <= 2:
        return min(slider_alpha, 0.30)
    return slider_alpha

def run_retrieval(user_text: str | None, image_path: str | None) -> pd.DataFrame:
    # Auto-select mode:
    if user_text and image_path:
        eff_alpha = infer_alpha(user_text, alpha)
        res = rec.recommend_hybrid(user_text, image_path, alpha=eff_alpha, top_k=top_k)
    elif image_path:
        res = rec.recommend_by_image(image_path, top_k=top_k)  # CV
    elif user_text:
        res = rec.recommend_by_text(user_text, top_k=top_k)    # NLP
    else:
        return pd.DataFrame()
    return join_results(res, articles_df, attach_images=True)


# ------------------------ Header ------------------------
st.title("🛍️ Shopping GPT")
st.caption(
    "One input like ChatGPT — click **+** to attach an image (drop / **paste Ctrl+V** / browse). "
    "I’ll search and reply with grounded tips."
)


# ------------------------ Render prior messages ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["role"] == "user" and msg.get("image_path") and os.path.exists(msg["image_path"]):
            st.image(msg["image_path"], width=280)
        if msg["role"] == "assistant":
            results_df = records_to_df(msg.get("results_records", []))
            if not results_df.empty:
                render_results_grid(results_df, show_detail=show_detail)


# ------------------------ “+” ATTACH next to the chat box ------------------------
col_left, col_right = st.columns([0.08, 0.92])

with col_left:
    # Popover opens a tiny panel like ChatGPT’s attach menu
    pop = st.popover("➕", use_container_width=True)
    with pop:
        st.write("Attach image (drop / **paste Ctrl+V** / browse):")
        up = st.file_uploader(
            " ", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
            key=f"img_up_{st.session_state.uploader_key}"  # resettable key
        )
        if up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1].lower()) as tmp:
                tmp.write(up.read())
                st.session_state.attach_image_tmp = tmp.name
            st.success("Image attached for your next message.")

    # show a small chip if something is attached
    if st.session_state.attach_image_tmp:
        st.markdown(
            '<div class="attach-chip">📎 image attached <span style="opacity:0.6">• will send with next message</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Remove", key="remove_attach"):
            try:
                p = st.session_state.attach_image_tmp
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
            st.session_state.attach_image_tmp = None
            st.session_state.uploader_key += 1  # force-reset uploader state
            st.rerun()

with col_right:
    # main chat input (Enter to send)
    user_text = st.chat_input("Describe the look you want, or ask a styling question...")


# ------------------------ Handle a new turn ------------------------
if user_text is not None:
    # 1) User message with optional image
    st.session_state.messages.append({
        "role": "user",
        "text": user_text,
        "image_path": st.session_state.attach_image_tmp
    })

    # 2) Retrieval (CV/NLP/Hybrid depending on what you provided)
    results_df = run_retrieval(user_text.strip(), st.session_state.attach_image_tmp)

    # 3) Grounded LLM answer (pass short history if assistant supports it)
    history = [
        {"role": m["role"], "content": m["text"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ][-6:]
    try:
        assistant_text = assistant.answer(user_text, results_df, history=history)
    except TypeError:
        # Backward-compatible with earlier chatbot.py
        if results_df is None or results_df.empty:
            assistant_text = (
                "I couldn't find matching items yet. Try describing a product or attach a photo with the ➕ button."
            )
        else:
            assistant_text = assistant.answer(user_text, results_df)

    records = df_to_records_for_store(results_df) if (results_df is not None and not results_df.empty) else []

    # 4) Assistant message with result cards
    st.session_state.messages.append({
        "role": "assistant",
        "text": assistant_text,
        "results_records": records
    })

    # 5) Clear one-time attachment
    st.session_state.attach_image_tmp = None
    st.session_state.uploader_key += 1  # also reset uploader after send

    # 6) Rerun to render
    st.rerun()
