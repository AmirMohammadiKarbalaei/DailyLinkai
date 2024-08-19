# DailyLinkai

Your Personalised Daily News Recommender.

## Overview
------------

DailyLinkai is an advanced AI-powered news aggregation web app that redefines the way you consume news. By leveraging cutting-edge machine learning and natural language processing techniques, it curates a selection of articles tailored to your interests. The aim for the app is to continuously evolves with your preferences through smart learning, making irrelevant news a thing of the past.

## Key Features
------------

* **Personalised News Feed**: Receive a daily feed of news articles that align with your interests and preferences.
* **Semi-supervised Learning**: Our intelligent recommendation system automatically labels news articles from reputable sources, ensuring content accuracy and relevance.
* **LLM Embeddings & Cosine Similarity Scores**: Advanced technologies assess article similarity, identifying related news pieces that resonate with your interests.
* **Active Learning Suggestion Model**: The system learns from your daily interactions, analysing engagement and clickthrough rates to deliver increasingly accurate and personalised news suggestions.
* **Effortless Experience**: Say goodbye to overwhelming news feeds; our system handles the hard work of curating news tailored just for you.

## How it Works
----------------

1. **Initial Setup**: Provide explicit preferences to start receiving relevant article suggestions.
2. **User Interaction**: Engage with articles by reading and rating them, enabling the system to understand your preferences.
3. **AI-powered Analysis**: Our AI analyses your interactions and preferences to build a personalised profile.
4. **Iterative Learning**: The system continuously refines and optimises news suggestions based on your evolving interests.

## Benefits
------------

* **Effortless News Consumption**: Eliminate the clutter of irrelevant articles and focus on what matters to you.
* **Personalised News Curation**: Enjoy a daily news feed that is specifically tailored to your tastes and interests.
* **Discover New Topics**: Explore new areas of interest and stay informed on the topics you care about most.

## Technology Stack
--------------------

* **Frontend**: Streamlit
* **Backend**: Python, Faiss, Langid, Numpy, Pandas, Psycopg2
* **Database**: PostgreSQL

---

## Get Started
---------------

Experience the ease of personalised news curation with DailyLinkai. To begin using the current version of the app without needing login credentials or a database connection, follow these steps:

1. Clone the repository.
2. Install the dependencies listed in `requirements.txt`.
3. Navigate to the `Deployment_wo_db` directory.
4. Run the app with the command `streamlit run app.py`.

This version of DailyLinkai pulls today's news articles from BBC and recommends content based on your preferences as you interact with the displayed news.

## Contributing
------------

Contributions are welcome to DailyLinkai. If you're interested in contributing, please fork this repository
and submit a pull request.

### To Do List:
1. **Semi-supervised Learning**: Create a system to labels news articles from reputable sources, ensuring content accuracy and relevance. 
2. **Active Learning Suggestion Model**: Implement active learning from customer interaction data and create personalised customer profile.

## License
-------

DailyLinkai is licensed under the [MIT License](https://opensource.org/licenses/MIT).

-------

