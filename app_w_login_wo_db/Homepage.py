import os
import random
import re
import smtplib
from email.mime.text import MIMEText
from time import sleep

import streamlit as st
import yaml
from argon2 import PasswordHasher

from navigation import make_sidebar

ph = PasswordHasher()


def load_user_data(filepath="files/users.yaml"):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as file:
        return yaml.safe_load(file) or {}


def save_user_data(data, filepath="../files/users.yaml"):
    with open(filepath, "w") as file:
        yaml.dump(data, file)


# Function to validate email format
def validate_email(email):
    """
    Validates the format of an email address.

    Args:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


# Function to validate password strength
def validate_password(password):
    """
    Validates the strength of a password.

    Args:
        password (str): The password to be validated.

    Returns:
        bool: True if the password is at least 8 characters long, False otherwise.
    """
    return len(password) >= 8


# Hash a password
def hash_password(password):
    """
    Hashes a given password using the PasswordHasher instance.

    Args:
        password (str): The password to be hashed.

    Returns:
        str: The hashed password.
    """
    return ph.hash(password)


# Check password
def check_password(stored_password, provided_password):
    """
    Checks if the provided password matches the stored password.

    Args:
        stored_password (str): The password stored in the system.
        provided_password (str): The password provided by the user.

    Returns:
        bool: True if the passwords match, False otherwise.
    """
    try:
        ph.verify(stored_password, provided_password)
        return True
    except Exception as e:
        print(f"An error occurred during password check: {e}")
        return False


# Function to handle user sign-up
def signup(email, password):
    """
    Handles the user sign-up process.

    Args:
        email (str): The user's email address.
        password (str): The user's password.

    Returns:
        tuple: A boolean indicating whether the sign-up was successful, and a message describing the result.
    """
    if not validate_email(email):
        return False, "Invalid email format."
    elif not validate_password(password):
        return False, "Password should be at least 8 characters long."

    users = load_user_data()

    if email in users:
        return False, "User already exists."

    # Generate and send verification code
    verification_code = generate_verification_code()
    if not send_verification_email(email, verification_code):
        return False, "Failed to send verification email."

    # Temporarily store user data
    if "signup_email" not in st.session_state:
        st.session_state["signup_email"] = email
        st.session_state["signup_password"] = hash_password(password)
    st.session_state["verification_code"] = verification_code
    return True, "Verification code sent to your email."


# Function to handle user sign-in
def signin(email, password):
    """
    Handles the user sign-in process.

    Args:
        email (str): The user's email address.
        password (str): The user's password.

    Returns:
        tuple: A boolean indicating whether the sign-in was successful, and a message describing the result.
    """
    users = load_user_data()

    if email not in users:
        return False, "User does not exist."
    elif not check_password(users[email]["password"], password):
        return False, "Incorrect password."
    else:
        return True, "User signed in successfully."


def send_verification_email(email, code):
    """
    Sends a verification email to the specified email address.

    Args:
        email (str): The recipient's email address.
        code (str): The verification code to be sent.

    Returns:
        bool: True if the email is sent successfully, False otherwise.
    """
    sender_email = st.secrets["owner"]["email"]
    sender_password = st.secrets["owner"]["password"]
    subject = "Your Verification Code for DailyLinkai"
    body = f"Your code is: {code}."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

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


def save_verified_user(email, password):
    """
    Saves a verified user to the user database.

    Args:
        email (str): The user's email address.
        password (str): The user's password.

    Returns:
        None
    """

    users = load_user_data()
    users[email] = {"password": hash_password(password)}
    save_user_data(users)


# Function to generate verification code
def generate_verification_code():
    """
    Generates a random verification code.

    Returns:
        str: A 6-digit verification code as a string.
    """
    return str(random.randint(100000, 999999))


def main():
    make_sidebar()

    st.title("✨ DailyLinkai ✨")

    # Ensure `signup_step` is initialized
    if "signup_step" not in st.session_state:
        st.session_state["signup_step"] = 1

    with st.expander("Sign In", expanded=False):
        st.subheader("Sign In to your account")

        # Sign in form
        signin_email = st.text_input("Email", key="signin_email")
        signin_password = st.text_input(
            "Password", type="password", key="signin_password"
        )

        if st.button("Sign In"):
            success, message = signin(signin_email, signin_password)
            if success:
                st.success(message)
                st.session_state.logged_in = True
                st.session_state["user_email"] = (
                    signin_email  # Use a different key to avoid the exception
                )
                # st.success("Logged in successfully!")
                sleep(0.5)
                st.switch_page("pages/app.py")
            else:
                st.error(message)

    if st.session_state["signup_step"] == 1:
        with st.expander("Sign Up", expanded=False):
            st.subheader("Create a new account")

            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input(
                "Password", type="password", key="signup_password"
            )

            if st.button("Sign Up"):
                if not validate_email(signup_email):
                    st.error("Invalid email format.")
                elif not validate_password(signup_password):
                    st.error("Password should be at least 8 characters long.")
                else:
                    success, message = signup(signup_email, signup_password)
                    if success:
                        st.success(message)
                        st.session_state["signup_step"] = 2
                    else:
                        st.error(message)

    if st.session_state["signup_step"] == 2:
        st.subheader("Verify Email")

        verification_code = st.text_input("Verification Code", key="verification")

        if st.button("Verify"):
            if (
                "verification_code" in st.session_state
                and st.session_state["verification_code"] == verification_code
            ):
                hashed_password = st.session_state["signup_password"]
                save_verified_user(st.session_state["signup_email"], hashed_password)

                st.success("Account created and logged in successfully!")
                st.session_state.logged_in = True
                st.session_state["user_email"] = st.session_state["signup_email"]
                st.session_state["signup_step"] = 1
                st.session_state.pop("signup_email", None)
                st.session_state.pop("signup_password", None)
                st.session_state.pop("verification_code", None)
                sleep(0.5)
                st.switch_page("pages/app.py")
            else:
                st.error("Incorrect verification code.")


if __name__ == "__main__":
    main()
