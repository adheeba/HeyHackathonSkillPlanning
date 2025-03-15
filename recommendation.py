import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
st.set_page_config(
    page_title="Employee Recommendation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
.card {
    border: 2px solid #007bff; /* Blue border */
    background-color: #21C0FA; /* Light gray background */
    padding: 15px;
    border-radius: 8px; /* Rounded corners */
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Light shadow */
    margin-bottom: 15px;
    max-width: 300px; /* Optional: Controls width */
    }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.image("logo.jpeg", use_container_width=False)
    st.markdown("""
    ## About the tool
    AI Powered ERS is a Strategic Skill Planner tool which helps Work force planners
                in selecting the best employees from his company given the skill set needed.
    """)
    #st.button(":blue[Get Skills]")
    #st.button(":blue[Recommend Employees]")
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
with image:
  image = Image.open('hey_logo.jpg')
  st.image(image, width=300)
with title:
    st.title("🚀 :blue[AI-Powered ERS]")
    st.markdown("# Find the best employees based on your required skills!")
import csv


col11,col21,col31 = st.columns([0.5,1,0.3])
# User input for skills
with col21:
    #user_input = st.text_input("## Enter skills (comma-separated):", "")
    st.markdown("## Enter skills (comma-separated):")
    user_input = st.text_input("", "")

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
