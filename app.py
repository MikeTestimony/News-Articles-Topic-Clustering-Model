import streamlit as st
import joblib
import numpy as np

# 1. Initialize web page title configuration rules
st.set_page_config(page_title="News Article Topic Clustering", page_icon="📰", layout="centered")

st.title("Group 42: News Article Topic Clustering App 🚀")
st.write("Paste a raw news article below. Our trained Agglomerative Hierarchical model will determine its cluster stream automatically.")

# 2. Use caching to load the large model assets ONLY ONCE into memory
@st.cache_resource
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('agglomerative_model.pkl')
    return vectorizer, model

# Catch missing file paths safely before the app breaks
try:
    vectorizer, clustering_model = load_assets()
except FileNotFoundError:
    st.error("Error: Model asset configuration files (.pkl) not found in the current directory directory.")

# 3. Establish custom explicit string headers for your cluster labels
topic_mapping = {
    0: "Entertainment & Media 🎬",
    1: "Business & Finance 📈",
    2: "Technology & Innovation 💻",
    3: "Education & Academy 🎓",
    4: "Sports & Athletics ⚽",
    5: "General Outliers / Technology Noise 🔍",
    6: "General Outliers / Entertainment Noise 🔍"
}

# 4. Draw the frontend input box widget
user_article = st.text_area("Paste Article Content Here:", height=250, placeholder="Type or paste the news body text here...")

# 5. Process tracking when the user pushes the interface button
if st.button("Analyze & Assign Cluster"):
    if user_article.strip() == "":
        st.warning("Please enter some text before processing.")
    else:
        with st.spinner("Processing text and mapping cluster space..."):
            
            # Step 5a: Convert raw string text into the exact 5,000-dimensional TF-IDF vector matrix
            transformed_vector = vectorizer.transform([user_article]).toarray()
            
            # Step 5b: Send the raw array vector into the model to return the mapped cluster assignment index 
            predicted_cluster = clustering_model.fit_predict(transformed_vector)[0]
            
            # Step 5c: Extract the friendly name using the index
            topic_name = topic_mapping.get(predicted_cluster, f"Cluster Group {predicted_cluster}")
            
            # Step 5d: Output the results cleanly to the screen
            st.success(f"**Target Allocation Result:** {topic_name}")
            st.info("Note: This classification was determined completely via Unsupervised Hierarchical Text Layout Vector Analysis.")