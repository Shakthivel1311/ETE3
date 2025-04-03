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
st.set_page_config(
    page_title="CHRISPO '25 Analytics", 
    page_icon="🏆", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# Title and header
st.title("🏆 CHRISPO '25 Inter-College Tournament Analytics")
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

# Sidebar filters with enhanced complexity
st.sidebar.title("🔍 Filters")

# Function to create select all checkbox
def select_all_checkbox(filter_name, options, default=True):
    select_all = st.sidebar.checkbox(f"Select all {filter_name}", value=default)
    if select_all:
        selected = st.sidebar.multiselect(
            f"Select {filter_name}", 
            options, 
            options,
            key=f"{filter_name}_select"
        )
    else:
        selected = st.sidebar.multiselect(
            f"Select {filter_name}", 
            options, 
            key=f"{filter_name}_select"
        )
    return selected

# Sport filter with select all
with st.sidebar.expander("🎯 Sport Filters", expanded=True):
    selected_sports = select_all_checkbox("Sports", df['Sport'].unique())

# College filter with select all and search
with st.sidebar.expander("🏛️ College Filters", expanded=True):
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        select_all_colleges = st.checkbox("Select all Colleges", value=True)
    with col2:
        search_college = st.text_input("Search College", "")
    
    if search_college:
        college_options = [col for col in df['College'].unique() if search_college.lower() in col.lower()]
    else:
        college_options = df['College'].unique()
    
    if select_all_colleges:
        selected_colleges = st.multiselect(
            "Select Colleges", 
            college_options, 
            college_options,
            key="college_select"
        )
    else:
        selected_colleges = st.multiselect(
            "Select Colleges", 
            college_options,
            key="college_select"
        )

# State filter with select all and state-wise counts
with st.sidebar.expander("🗺️ State Filters", expanded=True):
    # Show state counts for reference
    state_counts = df['State'].value_counts()
    st.sidebar.markdown("**State Participation Counts:**")
    for state, count in state_counts.items():
        st.sidebar.markdown(f"- {state}: {count} participants")
    
    selected_states = select_all_checkbox("States", df['State'].unique())

# Date filter with date range and select all
with st.sidebar.expander("📅 Date Filters", expanded=True):
    # Convert dates to string for display
    date_options = df['Participation_Date'].dt.strftime('%Y-%m-%d').unique()
    
    # Date range selector
    min_date = df['Participation_Date'].min()
    max_date = df['Participation_Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter dates based on range
    if len(date_range) == 2:
        filtered_dates = [
            d for d in date_options 
            if pd.to_datetime(date_range[0]) <= pd.to_datetime(d) <= pd.to_datetime(date_range[1])
        ]
    else:
        filtered_dates = date_options
    
    # Date multi-select with select all
    selected_dates = select_all_checkbox("Dates in Range", filtered_dates)

# Additional advanced filters
with st.sidebar.expander("⚙️ Advanced Filters", expanded=False):
    # Age range slider
    age_min, age_max = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Gender filter
    gender_options = df['Gender'].unique()
    selected_genders = st.multiselect(
        "Select Genders",
        gender_options,
        gender_options
    )
    
    # Previous participation filter
    prev_participation = st.selectbox(
        "Previous Participation",
        ["All", "Yes", "No"]
    )
    
    # Rating filter
    min_rating, max_rating = st.slider(
        "Rating Range",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )

# Apply all filters
filtered_df = df[
    (df['Sport'].isin(selected_sports)) &
    (df['College'].isin(selected_colleges)) &
    (df['State'].isin(selected_states)) &
    (df['Participation_Date'].dt.strftime('%Y-%m-%d').isin(selected_dates)) &
    (df['Age'] >= age_min) &
    (df['Age'] <= age_max) &
    (df['Gender'].isin(selected_genders)) &
    (df['Rating'] >= min_rating) &
    (df['Rating'] <= max_rating)
]

# Additional filter for previous participation
if prev_participation == "Yes":
    filtered_df = filtered_df[filtered_df['Previous_Participation'] == True]
elif prev_participation == "No":
    filtered_df = filtered_df[filtered_df['Previous_Participation'] == False]

# Show filter summary in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Filters:**")
st.sidebar.markdown(f"- Sports: {len(selected_sports)} selected")
st.sidebar.markdown(f"- Colleges: {len(selected_colleges)} selected")
st.sidebar.markdown(f"- States: {len(selected_states)} selected")
st.sidebar.markdown(f"- Dates: {len(selected_dates)} selected")
st.sidebar.markdown(f"- Age Range: {age_min} to {age_max}")
st.sidebar.markdown(f"- Genders: {', '.join(selected_genders)}")
st.sidebar.markdown(f"- Previous Participation: {prev_participation}")
st.sidebar.markdown(f"- Rating Range: {min_rating} to {max_rating}")
st.sidebar.markdown(f"**Total Records:** {len(filtered_df)}")

# Task 2: Dashboard Development
st.header("📊 Participation Trends Analysis")

# Display filtered data with download option
st.subheader("🔍 Filtered Participation Data")
st.dataframe(filtered_df, height=300, use_container_width=True)

# Add download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='chrispo_filtered_data.csv',
    mime='text/csv'
)

# Visualizations
st.subheader("📈 Participation Metrics")

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Participants", len(filtered_df))
with col2:
    st.metric("Sports Represented", filtered_df['Sport'].nunique())
with col3:
    st.metric("Colleges Represented", filtered_df['College'].nunique())
with col4:
    st.metric("Average Rating", round(filtered_df['Rating'].mean(), 2))

# Main visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏅 Participation by Sport")
    sport_counts = filtered_df['Sport'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sport_counts.values, y=sport_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Participants")
    ax.set_ylabel("Sport")
    st.pyplot(fig)

with col2:
    st.subheader("🏛️ Participation by College (Top 10)")
    college_counts = filtered_df['College'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    college_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette("pastel"))
    ax.set_ylabel("")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🗺️ Participation by State")
    state_counts = filtered_df['State'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_counts.index, y=state_counts.values, palette="rocket", ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Participants")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col4:
    st.subheader("📅 Participation by Day")
    day_counts = filtered_df['Participation_Date'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    day_counts.plot(kind='line', marker='o', ax=ax, color='purple')
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Participants")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Additional visualization
st.subheader("📊 Age Distribution by Sport")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Sport', y='Age', palette="Set3", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Task 3: Text Analysis
st.header("💬 Participant Feedback Analysis")

# Word cloud for each sport
st.subheader("☁️ Feedback Word Clouds by Sport")

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
st.subheader("⭐ Feedback Rating Comparison by Sport")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Sport', y='Rating', palette="Set2", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Task 4: Image Processing
st.header("📸 Sports Image Gallery")

# Create sample images directory if it doesn't exist
if not os.path.exists("sports_images"):
    os.makedirs("sports_images")
    # Create some sample images (in a real app, these would be actual images)
    for i in range(5):
        img = Image.new('RGB', (300, 200), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(f"sports_images/day_{i+1}.jpg")

# Day-wise image gallery
st.subheader("📅 Day-wise Image Gallery")
selected_day_img = st.selectbox("Select Day", ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])

# Display image
img_path = f"sports_images/{selected_day_img.lower().replace(' ', '_')}.jpg"
if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, caption=f"{selected_day_img} Highlights", use_container_width=True)
else:
    st.warning("Image not found for the selected day.")

# Custom image processing
st.subheader("🎨 Custom Image Processing")

uploaded_file = st.file_uploader("Upload a sports-related image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, caption="Original", use_container_width=True)
    
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
        
        st.image(processed_img, caption=processing_option, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🏆 CHRISPO '25 - Inter-College Tournament Analytics Dashboard")
st.markdown("Developed by the Shakthivel")
