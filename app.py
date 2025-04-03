import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image, ImageFilter, ImageEnhance
import os

# Set page config
st.set_page_config(page_title="CHRISPO '25 Analytics", page_icon="🏆", layout="wide")

# Custom CSS for better styling


# Title and header
st.title("CHRISPO '25 Inter-College Tournament Analytics")
st.markdown("---")

# Task 1: Dataset Generation
@st.cache_data
def generate_dataset():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample data options
    sports_events = [
        "Basketball", "Football", "Cricket", "Badminton", "Table Tennis",
        "Volleyball", "Athletics", "Swimming", "Chess", "Tennis"
    ]
    
    colleges = [
        "St. Xavier's College", "Christ University", "Loyola College",
        "St. Joseph's College", "Madras Christian College",
        "Presidency College", "St. Stephen's College", "St. Aloysius College",
        "St. Thomas College", "St. Paul's College"
    ]
    
    states = ["Karnataka", "Tamil Nadu", "Kerala", "Andhra Pradesh", "Maharashtra"]
    
    # Generate dates (5 consecutive days)
    start_date = datetime(2025, 2, 10)
    dates = [start_date + timedelta(days=i) for i in range(5)]
    
    # Generate feedback phrases
    feedback_phrases = [
        "Great experience", "Well organized", "Could improve facilities",
        "Excellent competition", "Loved participating", "Needs better scheduling",
        "Amazing atmosphere", "Good referees", "Food could be better",
        "Transportation issues", "Best tournament ever", "Medals were beautiful",
        "Communication was clear", "Some delays in events", "Fantastic volunteers",
        "Accommodation was comfortable", "Looking forward to next year",
        "Scorekeeping needs improvement", "Perfect venue", "More water stations needed"
    ]
    
    # Generate 300 participants
    data = []
    for i in range(1, 301):
        sport = random.choice(sports_events)
        college = random.choice(colleges)
        state = random.choice(states)
        date = random.choice(dates)
        feedback = random.choice(feedback_phrases) + " for " + sport
        
        participant = {
            "Participant_ID": f"P{i:03d}",
            "Name": f"Participant {i}",
            "College": college,
            "State": state,
            "Sport": sport,
            "Participation_Date": date.strftime("%Y-%m-%d"),
            "Age": random.randint(18, 25),
            "Gender": random.choice(["Male", "Female", "Other"]),
            "Previous_Participation": random.choice([True, False]),
            "Feedback": feedback,
            "Rating": random.randint(1, 5)
        }
        data.append(participant)
    
    df = pd.DataFrame(data)
    return df

# Load or generate data
df = generate_dataset()

# Convert date to datetime for easier manipulation
df['Participation_Date'] = pd.to_datetime(df['Participation_Date'])

# Sidebar filters
st.sidebar.title("Filters")
selected_sports = st.sidebar.multiselect("Select Sports", df['Sport'].unique(), df['Sport'].unique())
selected_colleges = st.sidebar.multiselect("Select Colleges", df['College'].unique(), df['College'].unique())
selected_states = st.sidebar.multiselect("Select States", df['State'].unique(), df['State'].unique())
selected_dates = st.sidebar.multiselect("Select Dates", df['Participation_Date'].dt.strftime('%Y-%m-%d').unique(), 
                                       df['Participation_Date'].dt.strftime('%Y-%m-%d').unique())

# Apply filters
filtered_df = df[
    (df['Sport'].isin(selected_sports)) &
    (df['College'].isin(selected_colleges)) &
    (df['State'].isin(selected_states)) &
    (df['Participation_Date'].dt.strftime('%Y-%m-%d').isin(selected_dates))
]

# Task 2: Dashboard Development
st.header("Participation Trends Analysis")

# Display filtered data
st.subheader("Filtered Participation Data")
st.dataframe(filtered_df, height=300, use_container_width=True)

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Participation by Sport")
    sport_counts = filtered_df['Sport'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sport_counts.values, y=sport_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Participants")
    ax.set_ylabel("Sport")
    st.pyplot(fig)

with col2:
    st.subheader("Participation by College")
    college_counts = filtered_df['College'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    college_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Participation by State")
    state_counts = filtered_df['State'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_counts.index, y=state_counts.values, palette="rocket", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Participants")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col4:
    st.subheader("Participation by Day")
    day_counts = filtered_df['Participation_Date'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    day_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Participants")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Additional visualization
st.subheader("Age Distribution by Sport")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Sport', y='Age', palette="Set3", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Task 3: Text Analysis
st.header("Participant Feedback Analysis")

# Word cloud for each sport
st.subheader("Feedback Word Clouds by Sport")

selected_sport_wc = st.selectbox("Select Sport for Word Cloud", df['Sport'].unique())

# Generate word cloud
feedback_text = ' '.join(filtered_df[filtered_df['Sport'] == selected_sport_wc]['Feedback'])
if feedback_text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Feedback Word Cloud for {selected_sport_wc}', fontsize=16)
    st.pyplot(fig)
else:
    st.warning("No feedback available for the selected sport with current filters.")

# Feedback comparison
st.subheader("Feedback Rating Comparison by Sport")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Sport', y='Rating', palette="Set2", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Task 4: Image Processing
st.header("Sports Image Gallery")

# Create sample images directory if it doesn't exist
if not os.path.exists("sports_images"):
    os.makedirs("sports_images")
    # Create some sample images (in a real app, these would be actual images)
    for i in range(5):
        img = Image.new('RGB', (300, 200), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(f"sports_images/day_{i+1}.jpg")

# Day-wise image gallery
st.subheader("Day-wise Image Gallery")
selected_day_img = st.selectbox("Select Day", ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])

# Display image
img_path = f"sports_images/{selected_day_img.lower().replace(' ', '_')}.jpg"
if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, caption=f"{selected_day_img} Highlights", use_column_width=True)
else:
    st.warning("Image not found for the selected day.")

# Custom image processing
st.subheader("Custom Image Processing")

uploaded_file = st.file_uploader("Upload a sports-related image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, caption="Original", use_column_width=True)
    
    with col2:
        st.subheader("Processed Image")
        processing_option = st.selectbox("Select Processing Option", 
                                       ["Grayscale", "Blur", "Enhance Contrast", "Edge Enhance"])
        
        if processing_option == "Grayscale":
            processed_img = img.convert('L')
        elif processing_option == "Blur":
            processed_img = img.filter(ImageFilter.BLUR)
        elif processing_option == "Enhance Contrast":
            enhancer = ImageEnhance.Contrast(img)
            processed_img = enhancer.enhance(2.0)
        elif processing_option == "Edge Enhance":
            processed_img = img.filter(ImageFilter.EDGE_ENHANCE)
        
        st.image(processed_img, caption=processing_option, use_column_width=True)

# Footer
st.markdown("---")
st.markdown("### CHRISPO '25 - Inter-College Tournament Analytics Dashboard")
st.markdown("Developed by the Shakthivel")