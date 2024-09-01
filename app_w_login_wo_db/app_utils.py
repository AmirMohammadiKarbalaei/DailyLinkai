import asyncio
import json
import logging
import os
import pickle

import aiohttp
import langid
import numpy as np
import pandas as pd
import streamlit as st
import torch
from bs4 import BeautifulSoup
from lxml import etree
from sitemaps_utils import remove_elements
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# Define your asynchronous function

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


async def fetch_url(session, url, timeout):
    """
    Asynchronously fetches the content of a given URL using the provided session object.

    Args:
        session (aiohttp.ClientSession): The session object used to make the HTTP request.
        url (str): The URL to fetch.
        timeout (float, optional): The timeout value in seconds. Defaults to None.

    Returns:
        str: The content of the fetched URL, or None if an error occurs.
    """
    try:
        async with session.get(url, timeout=timeout) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None


async def request_sentences_from_urls_async_app(urls, timeout=20):
    """
    Asynchronously extracts article content from a list of URLs.

    Args:
        urls (pd.DataFrame): A DataFrame containing the URLs to extract content from.
        timeout (int, optional): The timeout value in seconds for each URL request. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted article content, including the URL, topic, title, and content.
    """
    articles_df = pd.DataFrame(columns=["url", "topic", "title", "content"])

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, url in enumerate(urls.Url, start=1):
            if (idx - 1) % 100 == 0:
                logging.info(
                    f"\nProcessing URL {((idx - 1)//100)+1}/{(len(urls)//100)+1}"
                )

            tasks.append(fetch_url(session, url, timeout))

        results = await asyncio.gather(*tasks)

        for idx, (url, result) in enumerate(zip(urls.Url, results), start=1):
            if result is None:
                continue

            try:
                tree = etree.HTML(result)
                article_element = tree.find(".//article")
                if article_element is not None:
                    outer_html = etree.tostring(article_element, encoding="unicode")
                    article_body = remove_elements(outer_html)
                    article = [
                        line for line in article_body.split("\n") if len(line) >= 40
                    ]
                    articles_df.loc[idx - 1] = (
                        urls["Url"][idx - 1],
                        urls["Topic"][idx - 1],
                        urls["Title"][idx - 1],
                        " ".join(article),
                    )
                else:
                    # If no <article> element is found, try using BeautifulSoup with the specific ID
                    soup = BeautifulSoup(result, "html.parser")
                    article_id = (
                        "main-content"  # Replace with the actual ID you are targeting
                    )
                    article_element = soup.find(id=article_id)
                    if article_element:
                        article_body = remove_elements(str(article_element))
                        article = [
                            line for line in article_body.split("\n") if len(line) >= 40
                        ]
                        articles_df.loc[idx - 1] = (
                            urls["Url"][idx - 1],
                            urls["Topic"][idx - 1],
                            urls["Title"][idx - 1],
                            " ".join(article),
                        )
                    else:
                        logging.warning(
                            f"No article content found on the page with ID {article_id}."
                        )
            except Exception as e:
                logging.error(
                    f"Error extracting article content from {url}: error: {e}"
                )

    articles_df = (
        articles_df[
            (~articles_df["title"].str.contains("weekly round-up", case=False))
            & (articles_df["title"] != "One-minute World News")
        ]
        .drop_duplicates(subset="title")
        .reset_index(drop=True)
    )

    return articles_df


@st.cache_data
def collect_embed_content(df):
    """
    Collects and embeds news content from a given DataFrame.

    This function takes a DataFrame as input, removes duplicates based on the 'Title' column,
    and then fetches the news content using the `articles` function. It then embeds the news
    content using a pre-trained DPR Question Encoder model and stores the embeddings in the
    DataFrame.

    Args:
        df (DataFrame): A DataFrame containing news articles with 'Title' and 'content' columns.

    Returns:
        DataFrame: The input DataFrame with an additional 'embedding' column containing the
            embedded news content.
    """
    with st.spinner("Fetching news content"):
        df = df.drop_duplicates().reset_index(drop=True)
        collected_df = asyncio.run(articles(df, timeout=10))

        # st.write("bbc_news:",len(bbc_news.items()),"df:",len(df))

        # st.write(df)
        # for title, content in bbc_news.items():
        #     mask =  df['Title'] == title
        #     df.loc[mask, 'content'] = content
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(device)
        article_main_body = list(collected_df.content)

    with st.spinner("Embedding news content"):
        # Initialize progress bar
        progress_text = "Embedding Articles. This may take a few minutes. Please wait."
        my_bar = st.progress(0, text=progress_text)

        embeddings = []
        total_articles = len(article_main_body)

        for i, data in enumerate(article_main_body):
            # Tokenize with padding
            inputs = tokenizer(
                "".join(data),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(device)  # Move inputs to GPU
            with torch.no_grad():  # No need to track gradients during inference
                embedding = encoder(**inputs).pooler_output
            embeddings.append(embedding)

            # Update progress bar
            percent_complete = int((i + 1) / total_articles * 100)
            my_bar.progress(percent_complete, text=progress_text)

        # Convert embeddings tensor to numpy arrays
        embeddings_np = [embedding.cpu().numpy() for embedding in embeddings]

        # Convert embeddings to float32
        embeddings_np = [embedding.astype("float32") for embedding in embeddings_np]
        embeddings_np = np.array(embeddings_np).reshape(len(embeddings_np), 768)
        embeddings_list = [embedding for embedding in embeddings_np]
        # content_embedding = (list(bbc_news.values()), embeddings_np)
        # st.write("embeddings_np:",len(embeddings_np),"df:",len(df))

        collected_df["embedding"] = embeddings_list

        st.write("Content has been embedded!")
        my_bar.empty()

    return collected_df


# Function to load embeddings
@st.cache_resource
def load_embeddings():
    """
    Load the content embeddings from a pickle file.

    Returns:
        embeddings (list): A list of numpy arrays representing the content embeddings.
    """
    file_path = "content_embedding.pkl"

    with open(file_path, "rb") as file:
        embeddings = pickle.load(file)

    return embeddings


# Function to initialize interactions
# @st.cache_resource
def initialize_interactions():
    """
    Initializes a dictionary to track interactions with news articles.

    Returns:
        dict: A dictionary where each key is a news article ID and each value is another dictionary containing 'upvotes' and 'downvotes' counts.
    """
    return {"liked": [], "disliked": []}


# Function to track interactions
def track_interaction(interactions, news_id, action):
    """
    Tracks a user's interaction with a news article.

    Args:
        interactions (dict): A dictionary of news article interactions, where each key is a news article ID and each value is another dictionary containing 'upvotes' and 'downvotes' counts.
        news_id (str): The ID of the news article being interacted with.
        action (str): The type of interaction, either 'Upvoted' or 'Downvoted'.

    Returns:
        None
    """
    if action == "Upvoted":
        interactions["liked"].append(news_id)
    elif action == "Downvoted":
        interactions["disliked"].append(news_id)
    print(f"User interacted with news article {news_id} - {action}")


def streamlit_print_topic_counts(data_frame: pd.DataFrame, section_name: str):
    """
    Prints the count of each topic in the given DataFrame along with the total count.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the topic column.
        section_name (str): The name of the section being printed.

    Returns:
        None
    """
    st.subheader(section_name)
    topic_counts = data_frame["topic"].value_counts()
    st.write(topic_counts)
    st.write(f"Total: {topic_counts.sum()}")


# Main function for Streamlit app
st.cache_resource


async def articles(urls, timeout=20):
    """
    Asynchronously extracts article content from a list of URLs.

    Args:
        urls (pd.DataFrame): A DataFrame containing the URLs to extract content from.
        timeout (int, optional): The timeout value in seconds for each URL request. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted article content, including the URL, topic, title, and content.
    """

    articles = await request_sentences_from_urls_async_app(urls, timeout)
    return articles


def render_news_item(index, title, url):
    """
    Renders a news item with title, link, and upvote/downvote buttons.

    Args:
        index (int): The index of the news item in the list of news items.
        title (str): The title of the news item.
        url (str): The URL of the news item.

    Returns:
        None
    """

    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <h4>{title}</h4>
            <p><a href="{url}" target="_blank">Read more...</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Upvote", key=f"upvote_{index}"):
            track_interaction(
                st.session_state.session_user_interactions, index, "Upvoted"
            )
    with col2:
        if st.button("👎 Downvote", key=f"downvote_{index}"):
            track_interaction(
                st.session_state.session_user_interactions, index, "Downvoted"
            )


def save_interactions(interactions):
    """Save interactions to a JSON file."""
    with open(st.session_state.interaction_file, "w") as f:
        json.dump(interactions, f, indent=4)


def is_english_sentence(sentence: str):
    """
    This function takes a sentence as input and uses the langid
    library to classify the language of the sentence and returns True
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == "en"


def load_interactions():
    """
    Loads the user's interactions from a JSON file. If the file does not exist or
    there is an error loading the file, it returns None. Otherwise, it returns
    the loaded dictionary of interactions.
    """
    if os.path.exists(st.session_state.interaction_file):
        with open(st.session_state.interaction_file, "r") as f:
            try:
                return json.load(f)
            except:
                return None
    else:
        return None


def db_embeddings():
    """
    Retrieves articles from the database where the last modified date matches today's date.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved articles with columns 'id', 'date', 'title', 'url', 'topic', and 'embedding'.
    """

    try:
        # Attempt to read the CSV file
        articles = pd.read_csv("files/articles.csv", index_col=0)

        # If the DataFrame is empty, return an empty DataFrame
        if articles.empty:
            return None

        # If there are articles, filter by today's date and return the result
        articles_df = pd.DataFrame(articles)
        return articles_df  # [articles["date"] == today]

    except pd.errors.EmptyDataError:
        # If the CSV is empty or not found, return an empty DataFrame
        return None


def reset_sidebar():
    """
    Resets the state of the sidebar by setting the 'sidebar_reset' key in the session state to True.

    Parameters:
        None

    Returns:
        None
    """
    st.session_state.sidebar_reset = True
