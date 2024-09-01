from datetime import date

import faiss
import nest_asyncio
import numpy as np
import pandas as pd
import streamlit as st
from app_utils import (
    collect_embed_content,
    db_embeddings,
    initialize_interactions,
    load_interactions,
    render_news_item,
    reset_sidebar,
    streamlit_print_topic_counts,
)
from sitemaps_utils import Extract_todays_urls_from_sitemaps, process_news_data

from navigation import make_sidebar

nest_asyncio.apply()

st.session_state.interaction_file = "files/interactions.json"


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


def main():
    make_sidebar()

    # Load interactions from file
    loaded_inter = load_interactions()
    if loaded_inter is not None:
        if st.session_state["user_email"] in loaded_inter:
            st.session_state.all_time_current_user_interactions = loaded_inter[
                st.session_state["user_email"]
            ]
    if "session_user_interactions" not in st.session_state:
        st.session_state.session_user_interactions = initialize_interactions()
    if "displayed_news" not in st.session_state:
        st.session_state.displayed_news = 5
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
        todays_atricles = collect_embed_content(df_BBC)
        today = date.today()
        todays_atricles["date"] = today
        db = db_embeddings()
        if db is None:
            todays_atricles.to_csv("files/articles.csv")
            start = 0
            end = len(todays_atricles)
        else:
            db = db.drop_duplicates().reset_index(drop=True)
            start = len(db) + 1
            end = len(db) + len(todays_atricles)
            df = pd.concat([db, todays_atricles])
            df = df.drop_duplicates(subset="title").reset_index(drop=True)
            df.to_csv("files/articles.csv")

        status.update(label="Download complete!", state="complete", expanded=False)
    st.write("Today's Topics:")
    streamlit_print_topic_counts(todays_atricles, "BBC")
    preferences = st.multiselect(
        "What are your favorite Topics", todays_atricles.topic.unique()
    )
    selected_topic = st.radio(
        "Select a topic to show:", list(preferences), horizontal=True
    )
    if len(preferences) > 0:
        st.subheader(f"Latest {selected_topic} News")
    else:
        st.subheader("Please choose your preferences")

    selected_news = todays_atricles[todays_atricles["topic"] == selected_topic]
    for index, row in selected_news.head(st.session_state.displayed_news).iterrows():
        render_news_item(index, row["title"], row["url"])
    if st.session_state.displayed_news < len(selected_news):
        if st.button("Load More"):
            st.session_state.displayed_news += 5
    st.write(st.session_state.session_user_interactions)
    # most_upvoted = sorted(user_interactions["liked"], reverse=True)

    # counter = Counter(
    #     st.session_state.session_user_interactions[st.session_state.user_email]["liked"]
    # )

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

    user_email = st.session_state.user_email

    if loaded_inter is not None:
        if user_email in loaded_inter.keys():
            loaded_inter[user_email]["liked"] += [
                todays_atricles.title[x]
                for x in st.session_state.session_user_interactions["liked"]
            ]
            loaded_inter[user_email]["disliked"] += [
                todays_atricles.title[x]
                for x in st.session_state.session_user_interactions["disliked"]
            ]
        else:
            loaded_inter[user_email] = st.session_state.session_user_interactions
    else:
        loaded_inter = st.session_state.session_user_interactions

    st.session_state.interactions = loaded_inter

    for news_id in st.session_state.session_user_interactions["liked"]:
        for i in range(1, k):
            # if news_id < len(todays_atricles):
            news_row = todays_atricles.iloc[I[news_id][i]]

            if news_row.topic in preferences:
                if Suggestions:
                    st.sidebar.header("Suggestions:")
                    Suggestions = False

                st.sidebar.markdown(
                    title_style.format(title=news_row["title"]),
                    unsafe_allow_html=True,
                )
                st.sidebar.markdown(
                    link_style.format(url=news_row["url"]), unsafe_allow_html=True
                )
            else:
                if other_topics_printed:
                    st.sidebar.subheader("Similar articles from other topics:")
                    other_topics_printed = False

                st.sidebar.markdown(
                    title_style.format(title=news_row["title"]),
                    unsafe_allow_html=True,
                )
                st.sidebar.markdown(
                    link_style.format(url=news_row["url"]), unsafe_allow_html=True
                )
            # else:
            #     continue

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
