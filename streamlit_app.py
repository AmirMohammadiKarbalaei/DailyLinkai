import streamlit as st
from time import sleep
from navigation import make_sidebar
import streamlit as st
import psycopg2
from argon2 import PasswordHasher
import re
import smtplib
from email.mime.text import MIMEText
import random



ph = PasswordHasher()

# Function to validate email format
def validate_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

# Function to validate password strength
def validate_password(password):
    return len(password) >= 8

# Hash a password
def hash_password(password):
    return ph.hash(password)

# Check password
def check_password(stored_password, provided_password):
    try:
        ph.verify(stored_password, provided_password)
        return True
    except:
        return False

# Function to handle user sign-up
def signup(email, password):
    if not validate_email(email):
        return False, "Invalid email format."
    elif not validate_password(password):
        return False, "Password should be at least 8 characters long."

    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    if cursor.fetchone() is not None:
        cursor.close()
        connection.close()
        return False, "User already exists."

    # Generate and send verification code
    verification_code = generate_verification_code()
    if not send_verification_email(email, verification_code):
        return False, "Failed to send verification email."

    # Store the user data temporarily
    if "signup_email" not in st.session_state:
        st.session_state['signup_email'] = email
        st.session_state['signup_password'] = password
    st.session_state['verification_code'] = verification_code
    return True, "Verification code sent to your email."
# Function to handle user sign-in
def signin(email, password):
    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
    cursor = connection.cursor()
    
    cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if user is None:
        return False, "User does not exist."
    elif not check_password(user[0], password):
        return False, "Incorrect password."
    else:
        return True, "User signed in successfully."

def send_verification_email(email, code):
    sender_email = st.secrets["owner"]["email"]
    sender_password = st.secrets["owner"]["password"]
    subject = "Your Verification Code for DailyLinkai"
    body = f"Your code is: {code}."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    try:
        smtp_server, smtp_port = "smtp.gmail.com", 587
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [email], msg.as_string())
        server.quit()
        return True
    except ValueError as ve:
        print(f"Error: {ve}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Function to generate verification code
def generate_verification_code():
    return str(random.randint(100000, 999999))
make_sidebar()





st.title("✨ DailyLinkai ✨")

# Ensure `signup_step` is initialized
if 'signup_step' not in st.session_state:
    st.session_state['signup_step'] = 1

with st.expander("Sign In", expanded=False):
    st.subheader("Sign In to your account")

    # Sign in form
    signin_email = st.text_input("Email", key="signin_email")
    signin_password = st.text_input("Password", type='password', key="signin_password")

    if st.button("Sign In"):
        success, message = signin(signin_email, signin_password)
        if success:
            st.success(message)
            st.session_state.logged_in = True
            st.session_state['user_email'] = signin_email  # Use a different key to avoid the exception
            #st.success("Logged in successfully!")
            sleep(0.5)
            st.switch_page("pages/app.py")
        else:
            st.error(message)

if st.session_state['signup_step'] == 1:
    with st.expander("Sign Up", expanded=False):
        st.subheader("Create a new account")

        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type='password', key="signup_password")

        if st.button("Sign Up"):
            if not validate_email(signup_email):
                st.error("Invalid email format.")
            elif not validate_password(signup_password):
                st.error("Password should be at least 8 characters long.")
            else:
                success, message = signup(signup_email, signup_password)
                if success:
                    st.success(message)
                    st.session_state['signup_step'] = 2
                else:
                    st.error(message)

if st.session_state['signup_step'] == 2:
    st.subheader("Verify Email")

    verification_code = st.text_input("Verification Code", key="verification")


    if st.button("Verify"):
        if verification_code == st.session_state['verification_code']:
            hashed_password = hash_password(st.session_state['signup_password'])

            connection = psycopg2.connect(
                dbname=st.secrets["database"]["dbname"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"],
                host=st.secrets["database"]["host"],
                port=st.secrets["database"]["port"]
            )
            cursor = connection.cursor()
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (st.session_state['signup_email'], hashed_password))
            connection.commit()
            cursor.close()
            connection.close()

            st.success("Account created and logged in successfully!")
            st.session_state.logged_in = True
            st.session_state['user_email'] = st.session_state['signup_email']
            st.session_state['signup_step'] = 1
            st.session_state.pop('signup_email', None)
            st.session_state.pop('signup_password', None)
            st.session_state.pop('verification_code', None)
            sleep(0.5)
            st.switch_page("pages/app.py")
        else:
            st.error("Incorrect verification code.")

