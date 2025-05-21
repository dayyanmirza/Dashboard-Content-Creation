import streamlit as st
import pandas as pd
from pathlib import Path
import re
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "TikTok"

# Helper functions
def parse_comments_file(path: Path) -> pd.DataFrame:
    rows = []
    current = {}
    date_path = re.compile(r"^Date:\s+(.*) UTC$")
    comment_path = re.compile(r"^Comment:\s+(.*)$")

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # match date 
            m = date_path.match(line)
            if m:
                current["date"] = pd.to_datetime(m.group(1))
                continue
            # match comment
            m = comment_path.match(line)
            if m:
                current["comment"] = m.group(1)
                rows.append(current.copy())
                current.clear()
                continue 

    return pd.DataFrame(rows)


def parse_favorite_effects_file(path: Path) -> pd.DataFrame:
    """Parse Favorite Effects.txt into a DataFrame with columns: date, effect_link."""
    rows = []
    current = {}
    date_path = re.compile(r"^Date:\s+(.*) UTC$")
    effect_path = re.compile(r"^Effect Link:\s+(.*)$")

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # match date 
            m = date_path.match(line)
            if m:
                current["date"] = pd.to_datetime(m.group(1))
                continue
            # match effect link
            m = effect_path.match(line)
            if m:
                current["effect_link"] = m.group(1)
                rows.append(current.copy())
                current.clear()
                continue 

    return pd.DataFrame(rows)



def analyse_sentiment(texts):
    model = get_sentiment_model()
    results = []
    for text in texts:
        res = model(text, truncation=True)
        results.extend(res)
    label_map = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}
    return [label_map.get(r["label"], "Neutral") for r in results]


@st.cache_data # https://docs.streamlit.io/develop/concepts/architecture/caching
def load_comments() -> pd.DataFrame:
    return parse_comments_file(DATA_DIR / "Comments.txt")

def load_effects() -> pd.DataFrame:
    return parse_favorite_effects_file(DATA_DIR / "Favorite Effects.txt")

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=1)


# UI
st.set_page_config(page_title="Creator Dashboard", layout="wide")
st.title("Creator Insights Dashboard - TikTok")
st.markdown(
    """
    This is a simple dashboard to visualize TikTok comments.
    The data is parsed from the `Comments.txt` file in the `TikTok` directory.
    """
)

comments_df = load_comments()
effects_df = load_effects()

left, right = st.columns(2)
left.metric("Total Comments", len(comments_df))
right.metric("Earliest comment", str(comments_df["date"].min().date()))

st.subheader("All comments")
st.dataframe(comments_df, use_container_width=True)

st.subheader("Favorite Effects")
effects_df = effects_df.rename(columns={"effect_name": "Effect Name"})
st.dataframe(effects_df)


if st.button("Analyse Sentiment on Comments"):
    with st.spinner("Running sentiment analysis..."):
        comments_df["sentiment"] = analyse_sentiment(comments_df["comment"].tolist())
    st.success("Sentiment analysis complete.")
    
    # Show comments with sentiments
    st.subheader("Comments with Sentiment")
    st.dataframe(comments_df, use_container_width=True)

    # Sentiment distribution bar chart
    sentiment_counts = comments_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

