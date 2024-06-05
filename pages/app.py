import streamlit as st
import pandas as pd
from collections import defaultdict
import faiss
import numpy as np
import langid
from datetime import date
import psycopg2
from navigation import make_sidebar





def is_english_sentence(sentence:str):
    """
    This function takes a sentence as input and uses the langid 
    library to classify the language of the sentence and returns True 
    if the sentence is in english.
    """
    lang, confidence = langid.classify(sentence)
    return lang == 'en'
def initialize_interactions():
    return defaultdict(lambda: {'upvotes': 0, 'downvotes': 0})


def db_embeddigns():
    today = date.today()
    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM articles WHERE last_modified = %s", (today,) )
    articles = cursor.fetchall() 
    connection.commit()
    cursor.close()
    connection.close()
    articles_df = pd.DataFrame(articles,columns=["id","date","title","url","topic","embedding"])
    return articles_df

def streamlit_print_topic_counts(data_frame: pd.DataFrame):

    topic_counts = data_frame['topic'].value_counts()
    st.write(topic_counts)
    st.write(f"Total: {topic_counts.sum()}")

def track_interaction(interactions, news_id, action):
    if action == 'Upvoted':
        interactions[news_id]['upvotes'] += 1
    elif action == 'Downvoted':
        interactions[news_id]['downvotes'] += 1
    print(f"User interacted with news article {news_id} - {action}")
    print(interactions[news_id])

def reset_sidebar():
    st.session_state.sidebar_reset = True
def main():
    make_sidebar()
    if 'interactions' not in st.session_state:
        st.session_state['interactions'] = initialize_interactions()



# Access the interactions from session state
    interactions = st.session_state['interactions']

    
    st.title('✨ DailyLinkai ✨ ')
    
    with st.status("Collecting data...", expanded=True) as status:
        df = db_embeddigns()
        status.update(label="Download complete!", state="complete", expanded=False)
    st.write('Today\'s Topics:')
    streamlit_print_topic_counts(df)
    preferences = st.multiselect(
    "What are your favorite Topics",
    df.topic.unique())
    selected_topic = st.radio('Select a topic to show:', list(preferences),horizontal =True)
    if len(preferences)>0:
        st.subheader(f'Latest {selected_topic} News')
    else:
        st.subheader(f'Please choose your preferences')

    
    selected_news = df[df['topic'] == selected_topic]
    
    for index, row in selected_news.iterrows():
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <h4>{row['title']}</h4>
                <p><a href="{row['url']}" target="_blank">Read more...</a></p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Upvote", key=f"upvote_{index}"):
                track_interaction(interactions, index, 'Upvoted')
        with col2:
            if st.button("👎 Downvote", key=f"downvote_{index}"):
                track_interaction(interactions, index, 'Downvoted')

    
    # LOGO_URL_LARGE = "News-icon.jpg"
    # st.sidebar.image(LOGO_URL_LARGE)

    


    most_upvoted = sorted(interactions.items(), key=lambda x: x[1]['upvotes'], reverse=True)

    sorted_upvoted_idxs = [i[0] for idx, i in enumerate(most_upvoted) if i[1]["upvotes"] > 0]

    embeddings = [np.array(i) for i in df.embedding]
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
    # Perform a search to get the k nearest neighbors
    D, I = index.search(embeddings_np, k=k)
    other_topics_printed,Suggestions = True,True
    reset_sidebar()
    for news_id, counts in most_upvoted:
        # Check if the topic of the current news item is not in the user's preferences

        if counts['upvotes'] <= 0:
            continue
        
        for i in range(1, k):  # Start from 1 to skip the first neighbor (itself)
            # Fetch the news row corresponding to the current neighbor
            news_row = df.iloc[I[news_id][i]]
            print(I[news_id][i])
            


            if news_row.topic in preferences:
                if Suggestions:
                    st.sidebar.header('Suggestions:')
                    Suggestions = False
                
                st.sidebar.markdown(title_style.format(title=news_row['title']), unsafe_allow_html=True)
                st.sidebar.markdown(link_style.format(url=news_row['url']), unsafe_allow_html=True)
            else:
                if other_topics_printed:
                    st.sidebar.subheader("Similar articles from other topics:")
                    other_topics_printed = False

                st.sidebar.markdown(title_style.format(title=news_row['title']), unsafe_allow_html=True)
                st.sidebar.markdown(link_style.format(url=news_row['url']), unsafe_allow_html=True)

    if st.session_state['user_email']:
        email = st.session_state['user_email']


    # Step 1: Retrieve the current disliked articles for the user
    select_query = "SELECT liked, disliked FROM interactions WHERE email = %s;"
    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
    cursor = connection.cursor()
    cursor.execute(select_query, (email,))
    result = cursor.fetchone()
    # st.write(result)
   
    if len(interactions)>1:
        # liked_articles = " ".join([str(key) for key, value in interactions.items() if value['upvotes'] >= 1])
        # disliked_articles = " ".join([str(key) for key, value in interactions.items() if value['downvotes'] >= 1])
        liked_list = [key for key, value in interactions.items() if value['upvotes'] >= 1]
        disliked_list = [key for key, value in interactions.items() if value['downvotes'] >= 1]

        
        liked_articles_ids = [str(df.id[int(idx)]) for idx in liked_list]
        disliked_articles_ids = [str(df.id[int(idx)]) for idx in disliked_list]

        if result:
            liked, disliked = result[0], result[1]
            # st.write(liked.replace("{","").replace("}","").split(","))
            if len(liked) >0:
                liked_articles_ids +=  liked.replace("{","").replace("}","").split(",")
            if len(disliked) >0:
                disliked_articles_ids += disliked.replace("{","").replace("}","").split(",")
                # st.write(liked_list)
    




        if len(liked_list)>=1 and len(liked_list)>=1:
            if len(liked_list)>=2 or len(disliked_list)>=2:
                st.session_state['interactions'] = initialize_interactions()
                st.write("Saving to db")
                connection = psycopg2.connect(
                        dbname=st.secrets["database"]["dbname"],
                        user=st.secrets["database"]["user"],
                        password=st.secrets["database"]["password"],
                        host=st.secrets["database"]["host"],
                        port=st.secrets["database"]["port"]
                    )

                # Create a cursor to interact with the database
                cursor = connection.cursor()

                insert_query = """
                INSERT INTO interactions (email, liked, disliked)
                VALUES (%s, %s, %s)
                ON CONFLICT (email) 
                DO UPDATE SET 
                liked = EXCLUDED.liked,
                disliked = EXCLUDED.disliked;
                
                """

                cursor.execute(insert_query, (email, liked_articles_ids, disliked_articles_ids))


                # Commit the transaction and close the connection
                connection.commit()
                cursor.close()
                connection.close()
if __name__ == '__main__':
    main()
