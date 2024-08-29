import json
import os
from collections import defaultdict
from datetime import date
from collections import Counter

import faiss
import langid
import nest_asyncio
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from app_utils import (
    collect_embed_content,
    initialize_interactions,
    streamlit_print_topic_counts,
    track_interaction,
)
from sitemaps_utils import Extract_todays_urls_from_sitemaps, process_news_data

from navigation import make_sidebar

nest_asyncio.apply()

DATA_FILE = "interactions.json"


@st.cache_data
def fetch_and_process_news_data():
    """
    Fetches and processes today's news data from BBC news sitemaps.

    This function extracts URLs from the BBC news sitemaps, processes the news data,
    and returns a pandas DataFrame containing the processed news data.

    Parameters:
        None

    Returns:
        pd.DataFrame: A pandas DataFrame containing the processed news data.
    """
    with st.spinner("Fetching and processing todays news ..."):
        BBC_news_sitemaps = [
            "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
            "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
            "https://www.bbc.com/sitemaps/https-sitemap-com-news-3.xml",
        ]

        # sky_news_sitemaps = [
        #     "https://news.sky.com/sitemap/sitemap-news.xml",
        #     "https://www.skysports.com/sitemap/sitemap-news.xml",
        # ]

        namespaces = {
            "sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "news": "http://www.google.com/schemas/sitemap-news/0.9",
        }

        with st.spinner("Extracting URLs from BBC sitemaps..."):
            urls = {}
            urls["bbc"] = Extract_todays_urls_from_sitemaps(
                BBC_news_sitemaps, namespaces, "sitemap:lastmod"
            )
            # st.write(f"Found {len(urls['bbc'])} articles from BBC")

        # with st.spinner("Extracting URLs from Sky News sitemaps..."):
        #     urls["sky"] = Extract_todays_urls_from_sitemaps(sky_news_sitemaps, namespaces, 'news:publication_date')
        #     st.write(f"Found {len(urls['sky'])} URLs from Sky News")

        bbc_topics_to_drop = {"pidgin", "hausa", "swahili", "naidheachdan", "cymrufyw"}
        with st.spinner("Processing BBC A..."):
            df_BBC = process_news_data(urls, "bbc", bbc_topics_to_drop)

            # st.write("Data processing complete!")

        # Uncomment and complete the following if Sky News processing is required
        # sky_topics_to_drop = {"arabic", "urdu"}
        # with st.spinner("Processing Sky News data..."):
        #     df_Sky = process_news_data(urls, "sky", sky_topics_to_drop)
        #     st.write("Sky News data processing complete")
        # return pd.concat([df_BBC, df_Sky])

    return df_BBC.drop_duplicates("Title").reset_index(drop=True)


def reset_app():
    st.cache_data.clear()


st.markdown(
    """
    <style>
    .stButton > button {
        float: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_english_sentence(sentence: str):
    """
    This function takes a sentence as input and uses the langid
    library to classify the language of the sentence and returns True
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == "en"


def load_interactions():
    """Load interactions from a JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    else:
        return {}


def db_embeddings():
    """Simulate retrieving articles from a 'database'."""
    # For this example, we simulate with a fixed DataFrame
    today = date.today()

    try:
        # Attempt to read the CSV file
        articles = pd.read_csv("articles.csv")

        # If the DataFrame is empty, return an empty DataFrame
        if articles.empty:
            return pd.DataFrame()

        # If there are articles, filter by today's date and return the result
        articles_df = pd.DataFrame(articles)
        return articles_df  # [articles["date"] == today]

    except pd.errors.EmptyDataError:
        # If the CSV is empty or not found, return an empty DataFrame
        return pd.DataFrame()


def save_interactions(interactions):
    """Save interactions to a JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump(interactions, f, indent=4)


def reset_sidebar():
    """
    Resets the state of the sidebar by setting the 'sidebar_reset' key in the session state to True.

    Parameters:
        None

    Returns:
        None
    """
    st.session_state.sidebar_reset = True


def main():
    make_sidebar()

    # Load interactions from file
    st.session_state.interactions = load_interactions()
    user_interactions = st.session_state.interactions[st.session_state["user_email"]]

    if "interactions" not in st.session_state:
        st.session_state["interactions"] = initialize_interactions()

    st.title("✨ DailyLinkai ✨ ")
    st.button("Download Today's Articles", on_click=reset_app, key="Resetapp")

    with st.status("Collecting data...", expanded=True) as status:
        df_BBC = fetch_and_process_news_data()
        df_BBC = (
            df_BBC[
                (~df_BBC["Title"].str.contains("weekly round-up", case=False))
                & (df_BBC["Title"] != "One-minute World News")
            ]
            .drop_duplicates(subset="Title")
            .reset_index(drop=True)
        )
        df = db_embeddings()
        todays_atricles = collect_embed_content(df_BBC)
        df = pd.concat([df, todays_atricles], ignore_index=True)
        df.to_csv("articles.csv", index=True)
        status.update(label="Download complete!", state="complete", expanded=False)
    st.write("Today's Topics:")
    streamlit_print_topic_counts(todays_atricles, "BBC")
    preferences = st.multiselect("What are your favorite Topics", df.topic.unique())
    selected_topic = st.radio(
        "Select a topic to show:", list(preferences), horizontal=True
    )
    if len(preferences) > 0:
        st.subheader(f"Latest {selected_topic} News")
    else:
        st.subheader("Please choose your preferences")

    selected_news = todays_atricles[todays_atricles["topic"] == selected_topic]
    st.write(user_interactions)
    for index, row in selected_news.iterrows():
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <h4>{row['title']}</h4>
                <p><a href="{row['url']}" target="_blank">Read more...</a></p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Upvote", key=f"upvote_{index}"):
                track_interaction(user_interactions, index, "Upvoted")
        with col2:
            if st.button("👎 Downvote", key=f"downvote_{index}"):
                track_interaction(user_interactions, index, "Downvoted")

    # most_upvoted = sorted(user_interactions["liked"], reverse=True)
    counter = Counter(user_interactions["liked"])

    embeddings = [np.array(i) for i in todays_atricles.embedding]
    embeddings_np = np.array(embeddings)

    title_style = """
    <div style='font-size:15px; color:#white; margin-bottom:10px;'>
        {title}
    </div>
        """

    link_style = """
    <div style='margin-top:5px;'>
        <a href='{url}' style='color:#1f77b4; text-decoration:none; font-size:16px;'>
            Source
        </a>
    </div>"""

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    k = 3
    D, I = index.search(embeddings_np, k=k)
    other_topics_printed, Suggestions = True, True
    reset_sidebar()
    if len(user_interactions["liked"]) >= 3 or len(user_interactions["disliked"]) >= 3:
        st.session_state.interactions[st.session_state.user_email] = user_interactions
        save_interactions(st.session_state.interactions)
    for news_id, counts in counter.most_common():
        if counts <= 0:
            continue

        for i in range(1, k):
            news_row = todays_atricles.iloc[I[news_id][i]]

            if news_row.topic in preferences:
                if Suggestions:
                    st.sidebar.header("Suggestions:")
                    Suggestions = False

                st.sidebar.markdown(
                    title_style.format(title=news_row["title"]), unsafe_allow_html=True
                )
                st.sidebar.markdown(
                    link_style.format(url=news_row["url"]), unsafe_allow_html=True
                )
            else:
                if other_topics_printed:
                    st.sidebar.subheader("Similar articles from other topics:")
                    other_topics_printed = False

                st.sidebar.markdown(
                    title_style.format(title=news_row["title"]), unsafe_allow_html=True
                )
                st.sidebar.markdown(
                    link_style.format(url=news_row["url"]), unsafe_allow_html=True
                )

    # if st.session_state.get("user_email"):
    #     email = st.session_state["user_email"]

    #     # Retrieve the current disliked articles for the user
    #     user_data = interactions.get(email, {"liked": [], "disliked": []})
    #     liked_list = [
    #         key for key, value in interactions.items() if value["upvotes"] >= 1
    #     ]
    #     disliked_list = [
    #         key for key, value in interactions.items() if value["downvotes"] >= 1
    #     ]

    #     # Update user's liked and disliked lists
    #     user_data["liked"] = list(set(user_data["liked"] + liked_list))
    #     user_data["disliked"] = list(set(user_data["disliked"] + disliked_list))

    #     # Save updated interactions
    #     interactions[email] = user_data
    #     save_interactions(interactions)

    #     st.session_state["interactions"] = initialize_interactions()


if __name__ == "__main__":
    main()
