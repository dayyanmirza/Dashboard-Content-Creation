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


def parse_posts_file(path: Path) -> pd.DataFrame:
    """Parse Posts.txt into a DataFrame."""
    posts_data = []
    current_post = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line and current_post: # Handles blank lines between posts
                if "date" in current_post: # Ensure post has at least a date
                    posts_data.append(current_post)
                current_post = {}
                continue

            if line.startswith("Date:"):
                if current_post and "date" in current_post: # New post starts, save previous
                    posts_data.append(current_post)
                current_post = {} # Reset for new post
                current_post["date"] = pd.to_datetime(line.replace("Date:", "").replace("UTC", "").strip())
            elif line.startswith("Link:"):
                link = line.replace("Link:", "").strip()
                if "links" not in current_post:
                    current_post["links"] = [link]
                else:
                    # This is for additional links in photo mode posts if they are on new lines starting with "Link:"
                    # However, the provided file format usually lists multiple image links without "Link:" prefix for subsequent ones.
                    # This part might need adjustment based on exact multi-link format.
                    # For now, let's assume photo links appear one after another without "Link:" prefix.
                    current_post["links"].append(link) # Default to adding, review if structure is different
            elif "links" in current_post and current_post["links"] and ("http" in line or "https" in line) and ":" not in line:
                # Heuristic for subsequent photo links if they don't have a "Link:" prefix
                 current_post["links"].append(line)
            elif line.startswith("Like(s):"):
                current_post["likes"] = int(line.replace("Like(s):", "").strip())
            elif line.startswith("Who can view:"):
                current_post["who_can_view"] = line.replace("Who can view:", "").strip()
            elif line.startswith("Allow comments:"):
                current_post["allow_comments"] = line.replace("Allow comments:", "").strip().lower() == "yes"
            elif line.startswith("Allow stitches:"):
                current_post["allow_stitches"] = line.replace("Allow stitches:", "").strip().lower() == "yes"
            elif line.startswith("Allow duets:"):
                current_post["allow_duets"] = line.replace("Allow duets:", "").strip().lower() == "yes"
            elif line.startswith("Allow stickers:"):
                current_post["allow_stickers"] = line.replace("Allow stickers:", "").strip().lower() == "yes"
            elif line.startswith("Allow sharing to story:"):
                current_post["allow_sharing_to_story"] = line.replace("Allow sharing to story:", "").strip().lower() == "yes"
            elif line.startswith("Sound:"):
                current_post["sound"] = line.replace("Sound:", "").strip()
            elif line.startswith("Adds yours text:"):
                current_post["adds_yours_text"] = line.replace("Adds yours text:", "").strip()
            elif line.startswith("Title:"):
                current_post["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Content disclosure:"):
                 current_post["content_disclosure"] = line.replace("Content disclosure:", "").strip()


        if current_post and "date" in current_post: # Add the last post
            posts_data.append(current_post)

    df = pd.DataFrame(posts_data)
    # Determine post type
    if "links" in df.columns:
        df["post_type"] = df["links"].apply(lambda x: "Photo Carousel" if isinstance(x, list) and len(x) > 1 else "Video")
        df["link"] = df["links"].apply(lambda x: x[0] if isinstance(x, list) and x else (x if isinstance(x,str) else None) ) # take the first link as primary
    else:
        df["post_type"] = "Video" # Default if no 'links' column or it's malformed
        df["link"] = None

    # Reorder or select columns as needed
    desired_columns = [
        "date", "link", "likes", "sound", "post_type", "title", "adds_yours_text",
        "who_can_view", "allow_comments", "allow_stitches", "allow_duets",
        "allow_stickers", "allow_sharing_to_story", "content_disclosure", "links"
    ]
    # Ensure all desired columns exist, adding missing ones with None or default values
    for col in desired_columns:
        if col not in df.columns:
            df[col] = None # Or specific default like False for booleans, 0 for numbers etc.

    return df[desired_columns]




@st.cache_data # https://docs.streamlit.io/develop/concepts/architecture/caching
def load_comments() -> pd.DataFrame:
    return parse_comments_file(DATA_DIR / "Comments.txt")

def load_effects() -> pd.DataFrame:
    return parse_favorite_effects_file(DATA_DIR / "Favorite Effects.txt")

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=1)


@st.cache_data
def load_posts() -> pd.DataFrame:
    posts_file_path = DATA_DIR / "Posts.txt" 
    if posts_file_path.exists():
        return parse_posts_file(posts_file_path)
    else:
        st.error(f"Posts.txt not found at {posts_file_path}")
        return pd.DataFrame() # Return empty DataFrame if file not found


# UI
st.set_page_config(page_title="Creator Dashboard", layout="wide")
st.title("Creator Insights Dashboard - TikTok")
st.markdown(
    """
    This is a simple dashboard to visualise TikTok comments.
    The data is parsed from the `Comments.txt` file in the `TikTok` directory.
    """
)

comments_df = load_comments()
effects_df = load_effects()
posts_df = load_posts() 


left, right = st.columns(2)
left.metric("Total Comments", len(comments_df))
right.metric("Earliest comment", str(comments_df["date"].min().date()))

st.subheader("All comments")
st.dataframe(comments_df, use_container_width=True)

st.subheader("Favorite Effects")
effects_df = effects_df.rename(columns={"effect_name": "Effect Name"})
st.dataframe(effects_df)


st.subheader("Sentiment Analysis")
st.markdown(
    """
    This section allows you to perform sentiment analysis on the comments.
    The model used is `distilbert-base-uncased-finetuned-sst-2-english`.
    """
)

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



# Post analysis section
st.divider() # Adds a visual separator
st.header("Posts Analysis")

if not posts_df.empty:
    st.metric("Total Posts", len(posts_df))
    
    st.subheader("All Posts")
    st.dataframe(posts_df, use_container_width=True)

    st.subheader("Post Engagement Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Likes", f"{posts_df['likes'].sum():,}" if 'likes' in posts_df.columns else "N/A")
    col2.metric("Average Likes per Post", f"{posts_df['likes'].mean():.2f}" if 'likes' in posts_df.columns else "N/A")
    if 'likes' in posts_df.columns and not posts_df.empty :
         most_liked_post = posts_df.loc[posts_df['likes'].idxmax()]
         col3.metric("Most Likes on a Single Post", f"{most_liked_post['likes']:,} (on {str(most_liked_post['date'].date())})")
    else:
        col3.metric("Most Likes on a Single Post", "N/A")


    # Example Chart: Likes Over Time
    st.subheader("Likes Over Time")
    if 'date' in posts_df.columns and 'likes' in posts_df.columns:
        likes_over_time = posts_df.set_index('date')['likes']
        st.line_chart(likes_over_time)
    else:
        st.write("Date or Likes column missing for this chart.")

    # Example Chart: Posts by Type
    st.subheader("Posts by Type")
    if 'post_type' in posts_df.columns:
        post_type_counts = posts_df['post_type'].value_counts()
        st.bar_chart(post_type_counts)
    else:
        st.write("Post_type column missing for this chart.")

    # Example: Most Common Sounds
    st.subheader("Most Common Sounds (Top 10)")
    if 'sound' in posts_df.columns:
        common_sounds = posts_df['sound'].value_counts().nlargest(10)
        st.bar_chart(common_sounds)
    else:
        st.write("Sound column missing for this chart.")

else:
    st.warning("Posts data is empty or could not be loaded. Ensure 'Posts.txt' is in the 'TikTok' directory.")

