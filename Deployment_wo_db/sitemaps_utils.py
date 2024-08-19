import asyncio
import datetime
import json
import logging
import re
import string
import textwrap

import aiohttp
import langid
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)


# nltk.download('wordnet')
# nltk.download('stopwords')


def remove_elements(input_string: str) -> str:
    """
    This function takes an input string containing HTML elements and removes all HTML tags and their content.
    It also removes specific patterns such as "Published X hours ago" and "Image source".
    The function returns a cleaned string with the removed patterns and HTML tags replaced with spaces.

    Args:
        input_string (str): The input string containing HTML elements to be cleaned.

    Returns:
        str: The cleaned string with removed HTML tags and patterns.
    """
    # Parse the input string as HTML
    soup = BeautifulSoup(input_string, "html.parser")

    # Remove <script>, <style>, and <picture> tags and their content
    for script in soup(["script", "style", "picture"]):
        script.decompose()

    # Extract text from the parsed HTML
    cleaned_string = soup.get_text(separator="|", strip=True)
    splitted = cleaned_string.split("|")
    splitted_20 = [i for i in splitted if len(i) > 20]
    splitted_20 = ". ".join(splitted_20)

    # Define regular expressions to remove unwanted patterns
    patterns = [
        r"\bPublished.*?\bago\b",  # Matches "Published X hours ago"
        r"\bImage source\b",  # Matches "Image source, ..."
        r"\bImage caption\b",  # Matches "Image caption, ..."
        r"\bMedia caption\b",  # Matches "Media caption, ..."
        r"\bGetty Images\b",
        r"\bBBC Wales News\b",  # Matches "BBC Wales News"
        r"\bPublished\s\d{1,2}\s\w+\b",
        r"\bRelated Internet\b.*",
        r"\bBBC News Staff\b",
        r"\bFollow\s.*?\snews.*\b",
        r"\b\w+/\w+\b",
        r"Follow\sBBC.*",
        r'["\',]+',
        r"\s+",
    ]

    # Remove the matched patterns from the text
    for pattern in patterns:
        splitted_20 = re.sub(pattern, " ", splitted_20, flags=re.DOTALL)

    stuff_to_drop = [
        "fully signed version",
        "Use this form to ask your question:",
        "If you are reading this page and can t see the form you will need to visit the mobile version of the",
        "to submit your question or send them via email to",
        "YourQuestions@bbc.co.uk",
        "Please include your name age and location with any question you send in",
        "Related internet links",
        "This video can not be played",
        "To play this video you need to enable JavaScript in your browser",
        "Sign up for our morning newsletter and get BBC News in your inbox.",
        "Watch now on BBC iPlayer",
        "Listen now on BBC Sounds",
    ]
    for i in stuff_to_drop:
        splitted_20 = splitted_20.replace(i, "")

    cleaned_string = re.sub(r"\.\.|\.\s\.", ".", splitted_20)

    article = []
    for line in (i for i in cleaned_string.split("\n") if len(i) >= 10):
        article.append(line)

    return " ".join(article)


def clean_text(text):
    """
    This function cleans a given text by removing punctuation, stop words, and applying stemming and lemmatization.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # Download stopwords if not already downloaded
    stop_words = set(stopwords.words("english"))

    # Split the text into individual words
    words = text.split()

    # Remove stop words from the text
    words = [word for word in words if word.lower() not in stop_words]

    # Initialize the stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Apply stemming and lemmatization to each word
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]

    return " ".join(words)


def process_news_data(urls, source_name, topics_to_drop):
    """
    This function processes news data from a given source, filtering out non-English entries,
    live links, and unwanted topics. It takes in a dictionary of URLs, a source name, and a list
    of topics to drop. The function returns a pandas DataFrame containing the processed news data.

    Args:
        urls (dict): A dictionary containing news data from different sources.
        source_name (str): The name of the source to process.
        topics_to_drop (list): A list of topics to exclude from the processed data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the processed news data.
    """
    # Create DataFrame from the dictionary
    df_news = (
        pd.DataFrame.from_dict(urls[source_name], orient="index")
        .reset_index(drop=True)
        .rename(columns={"index": "Url"})
    )

    # Remove non-English entries
    df_news = df_news[df_news["Title"].apply(is_english_sentence)]

    df_news = df_news[~df_news["Url"].str.contains("live")]  # remove live links

    # Extract topics from 'Url'
    df_news["Topic"] = df_news["Url"].str.extract(r"com/([^/]+)/")[0]
    # edge_1 = df_news['Url'].str.extract(r'/([^/]+)/([^/]+)/([^/]+)/([^/]+)/')
    # df_news.drop(index=edge_1[(edge_1[1] == "live") | (edge_1[2] == "live") | (edge_1[3] == "live")].index, inplace=True)

    # Drop rows with unwanted topics
    df_news = df_news[~df_news["Topic"].isin(topics_to_drop)].reset_index(drop=True)

    # Print topic counts
    print_topic_counts(df_news, source_name)

    return df_news


def Extract_todays_urls_from_sitemaps(sitemaps: list, namespaces: dict, date_tag: str):
    """
    This function extracts the URLs and Last Modified
    and Title from XML sitemaps for today's date from given sitemaps

    Args:
    sitemaps (list): List of sitemap URLs to parse.
    namespaces (dict): Dictionary of XML namespaces.
    date_tag (str): The XML tag used to find the date (e.g., 'sitemap:lastmod' or 'news:publication_date').

    Returns:
    dict: Dictionary of URLs with their last modified date and title.
    """
    sitemap_data = {}
    today = datetime.date.today().isoformat()

    for sitemap in sitemaps:
        response = requests.get(sitemap)
        sitemap_xml = response.content

        root = etree.fromstring(sitemap_xml)

        urls = root.findall(".//sitemap:url", namespaces=namespaces)

        # Extract lastmod, URL, and news title for each <url> element with today's last modified date
        for url in urls:
            lastmod = url.findtext(f".//{date_tag}", namespaces=namespaces)
            loc = url.findtext(".//sitemap:loc", namespaces=namespaces)
            news_title = url.findtext(".//news:title", namespaces=namespaces)

            if lastmod and lastmod.startswith(today):
                sitemap_data[loc] = {
                    "index": loc,
                    "Last Modified": lastmod,
                    "Title": news_title,
                }

    return sitemap_data


def is_english_sentence(sentence: str):
    """
    Determines if a given sentence is written in English.

    Args:
        sentence (str): The sentence to be analyzed.

    Returns:
        bool: True if the sentence is written in English, False otherwise.
    """
    lang, confidence = langid.classify(sentence)
    return lang == "en"


def print_topic_counts(data_frame: pd.DataFrame, section_name: str):
    """
    Prints the count of each topic in the given DataFrame along with the total count.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the topic column.
        section_name (str): The name of the section being printed.

    Returns:
        None
    """
    print(f"------ {section_name} ------")
    topic_counts = data_frame["Topic"].value_counts()
    print(topic_counts.to_string())

    print(f"Total English articles: {topic_counts.sum()}")
    print()


def preprocess_title(title: str):
    """
    This function preprocesses a given title by removing special characters
    and converting it to lowercase.

    Args:
        title (str): The title to be preprocessed.

    Returns:
        str: The preprocessed title.
    """
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    return title.lower()


def calculate_similarity(
    title1: str, title2: str, model: str = "distilbert-base-nli-stsb-mean-tokens"
):
    """
    Calculates the similarity between two input titles using a pre-trained sentence embedding model.

    Args:
            title1 (str): The first title to compare.
            title2 (str): The second title to compare.
            model (str, optional): The pre-trained sentence embedding model to use. Defaults to "distilbert-base-nli-stsb-mean-tokens".

    Returns:
            float: A single similarity score between the two input titles.
    """

    embedding_title1 = model.encode([title1])
    embedding_title2 = model.encode([title2])
    similarity_score = cosine_similarity(embedding_title1, embedding_title2)[0][0]
    return similarity_score


# def request_sentences_from_urls(urls, timeout=20):
#     """
#     Extracts the main article content from a list of URLs using `requests` and `lxml`.
#     Returns a dictionary containing article bodies for each URL.
#     Handles fetch errors gracefully and returns None for failed URLs.
#     """

#     articles_dict = {}

#     for idx, url in enumerate(list(urls.Url), start=1):
#         if (idx-1)%10 ==0:
#             logging.info(f"\nProcessing URL {idx-1}/{len(urls)}")

#         try:
#             response = requests.get(url, timeout=timeout)
#             response.raise_for_status()  # Raise exception for HTTP errors
#         except requests.RequestException as e:
#             logging.error(f"Failed to fetch the web page: {e}")
#             continue
#         except TimeoutError:
#             logging.error(f"Timeout occurred while fetching URL: {url}")
#             continue

#         try:
#             tree = etree.HTML(response.content)
#             article_element = tree.find(".//article")

#             if article_element is not None:
#                 outer_html = etree.tostring(article_element, encoding='unicode')

#                 article_body = remove_elements(outer_html)

#                 article = []
#                 for line in (i for i in article_body.split("\n") if len(i) >= 40):
#                     article.append(line)
#                 articles_dict[urls["Title"][idx-1]] = article
#                 #logging.info("Article has been extracted")
#             else:
#                 logging.warning("No article content found on the page.")
#                 continue
#         except Exception as e:
#             logging.error(f"Error extracting article content: {e} URL: {url}")
#             continue


#     return articles_dict
async def fetch_url(session, url, timeout):
    """
    Asynchronously fetches the content of a given URL using the provided `session` object.

    Args:
        session (aiohttp.ClientSession): The session object used to make the HTTP request.
        url (str): The URL to fetch.
        timeout (float, optional): The timeout value in seconds. Defaults to None.

    Returns:
        str: The content of the fetched URL.

    Raises:
        aiohttp.ClientError: If there is an error with the HTTP request.
        asyncio.TimeoutError: If the request times out.

    """
    try:
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None


async def request_sentences_from_urls_async(urls, timeout=20):
    """
    Asynchronously requests and extracts article content from a list of URLs.

    Args:
        urls (pd.DataFrame): A DataFrame containing the URLs to fetch, with columns 'Url', 'Topic', and 'Title'.
        timeout (int, optional): The timeout value in seconds for each HTTP request. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted article content, with columns 'url', 'topic', 'title', and 'content'.
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

    return articles_df


def Text_wrap(Text: str, width: int):
    """
    Wraps the given text within the specified width for better printing visibility.

    Args:
        Text (str): The text to be wrapped.
        width (int): The desired width of the wrapped text.

    Returns:
        str: The wrapped text.

    """

    wrapped_string = textwrap.fill(Text, width=width)
    print(wrapped_string)
    return wrapped_string


def Save_to_json(file_name: str, data: any):
    """
    Saves the given data to a JSON file.

    Args:
            file_name (str): The name of the JSON file to be created.
            data (any): The data to be saved in the JSON file.

    Returns:
            None
    """
    with open(f"{file_name}.json", "w") as f:
        json.dump(data, f)
    print(f"File {file_name}.json has been saved")


def Open_json(file_name: str):
    """
    Opens and reads a JSON file.

    Args:
        file_name (str): The name of the JSON file to be opened.

    Returns:
        dict: The contents of the JSON file.
    """
    with open(file_name, "r") as f:
        file = json.load(f)
    return file
