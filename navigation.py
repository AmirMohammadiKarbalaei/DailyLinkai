from time import sleep

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages


def get_current_page_name():
    """
    Retrieves the name of the current page.

    Returns:
        str: The name of the current page.
    """
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = get_pages("")

    return pages[ctx.page_script_hash]["page_name"]


HORIZONTAL = "./media/DailyLinkai_horizontal.png"
ICON = "./media/D.png"


def make_sidebar():
    """
    Creates a sidebar for the application with a logo, title, and description.
    It also includes a login status check, providing a link to the main page if logged in,
    and a logout button. If not logged in, it redirects to the login page if attempting to access a secret page.

    Parameters:
        None

    Returns:
        None
    """
    with st.sidebar:
        st.logo(HORIZONTAL, icon_image=ICON)
        # st.image("./media/2.png",width=200)

        st.title(" 📰 Your News, Smarter 📰")

        st.write(
            "DailyLinkai 💎, Powered by cutting-edge AI, curates articles tailored to your interests, evolving with your preferences through smart learning. Say goodbye to irrelevant news and hello to a customized, effortless news experience. "
        )
        st.write(
            "Stay informed and explore new topics with DailyLinkai, your ultimate news companion!"
        )

        if st.session_state.get("logged_in", False):
            st.page_link("pages/app.py", label="Main", icon="🔒")

            st.write("")
            st.write("")

            if st.button("Log out"):
                logout()

        elif get_current_page_name() != "streamlit_app":
            # If anyone tries to access a secret page without being logged in,
            # redirect them to the login page
            st.switch_page("streamlit_app.py")


def logout():
    st.session_state.logged_in = False
    st.info("Logged out successfully!")
    sleep(0.5)
    st.switch_page("streamlit_app.py")
