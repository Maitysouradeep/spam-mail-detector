# Final-Year Project: Spam Mail Detection Web App
# Developed by: Souradeep Maity

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

# ----------------------------- DATABASE SETUP -----------------------------
conn = sqlite3.connect("user_data.db", check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    message TEXT,
    prediction TEXT,
    confidence REAL,
    timestamp TEXT
)''')
conn.commit()

# ----------------------------- AUTH UTILS -----------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed):
    return make_hashes(password) == hashed

def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, make_hashes(password)))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and check_hashes(password, user[1]):
        return user
    return None


# ----------------------------- STREAMLIT UI -----------------------------
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detection System")
st.markdown("Detect whether an email is **Spam** or **Ham** using a Machine Learning model.")

# ----------------------------- LOAD + TRAIN MODEL -----------------------------
@st.cache_resource
def load_model():
    data = pd.read_csv("mail_data.csv")
    data = data.where(pd.notnull(data), '')
    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1
    X = data['Message']
    Y = data['Category'].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    acc_train = accuracy_score(Y_train, model.predict(X_train_features))
    acc_test = accuracy_score(Y_test, model.predict(X_test_features))
    return model, vectorizer, acc_train, acc_test

model, vectorizer, acc_train, acc_test = load_model()

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.subheader("🔐 Optional Login")
menu = ["Home", "Login", "Register"]
choice = st.sidebar.selectbox("Navigation", menu)

if 'user' not in st.session_state:
    st.session_state.user = None
    # Show logout if user is logged in
if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()


if choice == "Register":
    st.sidebar.write("Create a new account")
    new_user = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Register"):
        add_user(new_user, new_password)
        st.sidebar.success("User registered. You can login now.")

elif choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        result = login_user(username, password)
        if result:
            st.session_state.user = username
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("Incorrect username/password")

# ----------------------------- MAIN APP -----------------------------
st.markdown(f"### Model Accuracy")
st.markdown(f"✅ Training Accuracy: `{acc_train*100:.2f}%`")
st.markdown(f"✅ Testing Accuracy: `{acc_test*100:.2f}%`")

st.markdown("---")
st.subheader("📥 Enter a New Email")
input_text = st.text_area("Type or paste an email message:")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        features = vectorizer.transform([input_text])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        label = "Ham" if pred == 1 else "Spam"
        color = "green" if label == "Ham" else "red"
        confidence = np.max(prob) * 100

        st.markdown(f"### 📊 Result: <span style='color:{color}; font-size:24px'><b>{label}</b></span>", unsafe_allow_html=True)
        st.markdown(f"Confidence: `{confidence:.2f}%`")

        # Save if logged in
        if st.session_state.user:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('INSERT INTO history (username, message, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?)',
                      (st.session_state.user, input_text, label, float(confidence), now))
            conn.commit()

# ----------------------------- BULK PREDICTION -----------------------------
st.markdown("---")
st.subheader("📤 Bulk Email Upload (CSV)")
uploaded_file = st.file_uploader("Upload CSV file with a 'Message' column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Message' in df.columns:
        features = vectorizer.transform(df['Message'])
        df['Prediction'] = model.predict(features)
        df['Prediction'] = df['Prediction'].apply(lambda x: "Ham" if x == 1 else "Spam")
        st.write(df[['Message', 'Prediction']])
        st.download_button("Download Results as CSV", df.to_csv(index=False).encode(), file_name="predictions.csv")
    else:
        st.error("The uploaded file must have a 'Message' column")

# ----------------------------- VIEW HISTORY -----------------------------
if st.session_state.user:
    st.markdown("---")
    st.subheader("📁 Your Prediction History")
    if 'deleted' not in st.session_state:
        st.session_state.deleted = False

    delete_success = False
    c.execute('SELECT id, message, prediction, confidence, timestamp FROM history WHERE username = ?', (st.session_state.user,))
    data = c.fetchall()

    if data:
        hist_df = pd.DataFrame(data, columns=["ID", "Message", "Prediction", "Confidence", "Time"])
        st.dataframe(hist_df)

        delete_col1, delete_col2 = st.columns([1, 2])
        with delete_col1:
            delete_id = st.text_input("Enter ID to delete a specific entry:")
            if st.button("Delete Entry") and delete_id.strip().isdigit():
                c.execute('DELETE FROM history WHERE id = ? AND username = ?', (int(delete_id.strip()), st.session_state.user))
                conn.commit()
                st.session_state.deleted = True
                st.experimental_rerun()

        with delete_col2:
            if st.button("🗑 Delete All History"):
                c.execute('DELETE FROM history WHERE username = ?', (st.session_state.user,))
                conn.commit()
                st.session_state.deleted = True
                st.experimental_rerun()
    else:
        st.info("No history yet.")

# ----------------------------- FOOTER -----------------------------
st.markdown("---")
st.caption("Developed by Souradeep Maity · Final Year Project · 2025")


