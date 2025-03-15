import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("telecom_employee_skills_dataset.csv")

df = load_data()

# Convert skills column into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Skills'])

# Streamlit UI
st.title("AI-Powered Employee Recommendation System")
st.write("Enter skills to find the best-matching employees.")

# User input for skills
user_input = st.text_input("Enter skills (comma-separated):", "")

if user_input:
    # Convert user input into vector
    input_vector = vectorizer.transform([user_input])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Get top 10 most relevant employees
    df["Similarity"] = similarity_scores
    recommendations = df.sort_values(by="Similarity", ascending=False).head(10)

    # Display results
    if not recommendations.empty:
        st.success(f"Top {len(recommendations)} recommended employees:")
        st.dataframe(recommendations[['Employee_ID', 'Name', 'Department', 'Job_Role', 'Skills', 'Similarity']])
    else:
        st.warning("No employees found with the specified skills.")
