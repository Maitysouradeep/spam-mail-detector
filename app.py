import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detection System")
st.markdown("This app uses Machine Learning to detect whether a message is **Spam** or **Ham** (Not Spam).")

# Load dataset
try:
    mail_data = pd.read_csv("mail_data (1).csv")  # ← Use exact filename
    mail_data = mail_data.where(pd.notnull(mail_data), '')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    mail_data['Category'] = mail_data['Category'].astype(int)
    st.success("✅ Dataset loaded successfully!")
    st.write(mail_data.head())
except Exception as e:
    st.error("❌ Could not load mail_data.csv file.")
    st.exception(e)
    st.stop()

# Prepare data
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# UI Input
st.header("🔍 Test a New Email")
user_input = st.text_area("✉️ Enter the email content below:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("✅ This is a **Ham** email.")
        else:
            st.error("🚫 This is a **Spam** email.")

        st.info(f"🔎 Confidence: {np.max(proba)*100:.2f}%")

st.markdown("---")
st.caption("Final Year Project · Souradeep Maity · ML with Streamlit")
