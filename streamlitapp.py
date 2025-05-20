import streamlit as st
import pandas as pd
from pathlib import Path
import re

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


@st.cache_data # https://docs.streamlit.io/develop/concepts/architecture/caching
def load_data() -> pd.DataFrame:
    return parse_comments_file(DATA_DIR / "Comments.txt")
    

# UI
st.set_page_config(page_title="Creator Dashboard", layout="wide")
st.title("Creator Insights Dashboard - TikTok")
st.markdown(
    """
    This is a simple dashboard to visualize TikTok comments.
    The data is parsed from the `Comments.txt` file in the `TikTok` directory.
    """
)

df = load_data()

left, right = st.columns(2)
left.metric("Total Comments", len(df))
right.metric("Earliest comment", str(df["date"].min().date()))

st.subheader("All comments")
st.dataframe(df, use_container_width=True)