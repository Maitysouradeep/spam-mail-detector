import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detection System")

st.markdown("This app detects whether an email is spam or not using Machine Learning.")

# Load CSV with error handling
try:
    mail_data = pd.read_csv("mail_data.csv")
    mail_data = mail_data.where(pd.notnull(mail_data), '')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    mail_data['Category'] = mail_data['Category'].astype(int)
    st.success("✅ Dataset loaded successfully!")
    st.write(mail_data.head())
except Exception as e:
    st.error("❌ Failed to load mail_data.csv file.")
    st.exception(e)
    st.stop()

# Model pipeline
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

# App UI
st.header("📬 Test a New Email")
input_text = st.text_area("Enter email message:")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        input_data = vectorizer.transform([input_text])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("✅ This is a **Ham** email.")
        else:
            st.error("🚫 This is a **Spam** email.")

        st.info(f"Confidence: {np.max(probability) * 100:.2f}%")

st.markdown("---")
st.caption("Project by Souradeep Maity · Final Year B.Tech")

