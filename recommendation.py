import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="'AI-Powered Employee Recommendation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("telecom_employee_skills_dataset.csv")

df = load_data()

# Convert skills column into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Skills'])

# Streamlit UI
image, title = st.columns([0.8,2])
with title:
    st.title("🚀 :blue[_AI-Powered_ _Employee_ _Recommendation_ _System_]")
    st.markdown("Find the best employees based on your required skills!")
import csv


col11,col21,col31 = st.columns([0.5,1,0.3])
# User input for skills
with col21:
    user_input = st.text_input("Enter skills (comma-separated):", "")

if user_input:
    # Convert user input into vector
    input_vector = vectorizer.transform([user_input])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Get top 10 most relevant employees
    df["Similarity"] = similarity_scores
    recommendations = df.sort_values(by="Similarity", ascending=False).head(10)
    with col21:
        st.subheader("Recommended Employees")
    with col21:
        col1, col2, col3 = st.columns(3)
        # Display results
        if not recommendations.empty:
            
            st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
            count = 0
            for _, row in recommendations.iterrows():
                if count % 3 == 0:
                    with col1:
                        st.markdown(f"""
                            <div class="card">
                                <h4>👤 {row['Name']}</h4>
                                <p><b>Department:</b> {row['Department']}</p>
                                <p><b>Role:</b> {row['Job_Role']}</p>
                                <p><b>Skills:</b> {row['Skills']}</p>
                                <p><b>Match Score:</b> {round(row['Similarity'] * 100, 2)}%</p>
                            </div>
                        """, unsafe_allow_html=True)

                if count % 3 == 1:
                    with col2:
                        st.markdown(f"""
                            <div class="card">
                                <h4>👤 {row['Name']}</h4>
                                <p><b>Department:</b> {row['Department']}</p>
                                <p><b>Role:</b> {row['Job_Role']}</p>
                                <p><b>Skills:</b> {row['Skills']}</p>
                                <p><b>Match Score:</b> {round(row['Similarity'] * 100, 2)}%</p>
                            </div>
                        """, unsafe_allow_html=True)

                if count % 3 == 2:
                    with col3:
                        st.markdown(f"""
                            <div class="card">
                                <h4>👤 {row['Name']}</h4>
                                <p><b>Department:</b> {row['Department']}</p>
                                <p><b>Role:</b> {row['Job_Role']}</p>
                                <p><b>Skills:</b> {row['Skills']}</p>
                                <p><b>Match Score:</b> {round(row['Similarity'] * 100, 2)}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                count = count + 1
            st.markdown("</div>", unsafe_allow_html=True)
            #st.success(f"Top {len(recommendations)} recommended employees:")
            #st.dataframe(recommendations[['Employee_ID', 'Name', 'Department', 'Job_Role', 'Skills', 'Similarity']])
        else:
            st.warning("No employees found with the specified skills.")
