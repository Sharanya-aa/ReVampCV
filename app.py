import time
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import regex as re
import textstat

# Document processing libraries
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.layout import LAParams
import docx

# NLP libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI imports
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai_client = None

def initialize_openai_client():
    """Initialize OpenAI client only once"""
    global openai_client
    if openai_client is not None:
        return openai_client
    
    try:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            # Only print once during initialization
            if 'openai_initialized' not in st.session_state:
                st.session_state.openai_initialized = True
        else:
            openai_client = None
            
    except Exception as e:
        openai_client = None
    
    return openai_client

# Initialize the client
openai_client = initialize_openai_client()

# Set page configuration
st.set_page_config(
    page_title="ReVamp CV",
    # page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App description
st.markdown("""
### AI-powered resume analyzer and enhancer

ReVamp CV helps job seekers optimize their resumes for specific job descriptions. 
Upload your resume and a job description to receive AI-powered suggestions, skill gap analysis, 
and tailored content improvements to increase your chances of getting past ATS systems and catching recruiters' attention.
The main goal of this project is to create a smart, AI-powered web application called
ReVamp CV. This tool helps job seekers customize and improve their resumes based on
specific job descriptions (JD). In today's competitive job market, generic resumes often do
not grab the attention of recruiters and are likely to be rejected by automated Applicant
Tracking Systems (ATS). This application tackles that issue by giving users a smart tool
to tailor their resumes to better fit what employers expect.
""")

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.warning("Please download the spaCy model by running: python -m spacy download en_core_web_sm")
    nlp = spacy.blank('en')
    # Add sentencizer component for sentence boundary detection
    nlp.add_pipe('sentencizer')

# Helper functions for file processing
def extract_text_from_pdf(pdf_file):
    try:
        # pdfminer.six provides better text extraction than PyPDF2
        # Configure layout parameters for better text extraction
        laparams = LAParams(
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
        
        # Change this line from pdf_extract_text to pdf_extract
        text = pdf_extract_text(pdf_file, laparams=laparams)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext == ".pdf":
        return extract_text_from_pdf(file)
    elif file_ext == ".docx":
        return extract_text_from_docx(file)
    elif file_ext == ".txt":
        return file.getvalue().decode("utf-8")
    else:
        st.error(f"Unsupported file format: {file_ext}")
        return ""

def calculate_similarity_score(resume_text, jd_text):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the texts to TF-IDF vectors
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100  # Convert to percentage
    except:
        st.error("Error calculating similarity score. Please check your resume and job description.")
        return 0

def summarize_text(text, max_length=500):
    """Generate a simple summary of the text using spaCy"""
    doc = nlp(text)
    
    # Get the sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # If there are fewer than 3 sentences, return the original text
    if len(sentences) <= 3:
        return text
    
    # Otherwise, take the first 3 sentences as a summary
    summary = " ".join(sentences[:3])
    
    # If the summary is still too long, truncate it
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
        
    return summary

def extract_key_points(text):
    """Extract key points from job description text"""
    doc = nlp(text)
    
    # Get the sentences and clean them
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # If there are very few sentences, use them all
    if len(sentences) <= 5:
        return [s for s in sentences if len(s) > 20]  # Filter out very short sentences
    
    # Otherwise, take key sentences based on position and keywords
    important_keywords = ['required', 'skill', 'qualification', 'experience', 'responsibilities', 
                         'duties', 'knowledge', 'abilities', 'proficient', 'familiar', 'degree',
                         'years', 'background', 'expertise', 'must have', 'should have', 'preferred']
    
    key_points = []
    
    # Always include the first sentence if it's substantial
    if sentences and len(sentences[0]) > 20:
        key_points.append(sentences[0])
    
    # Add sentences that contain important keywords
    for sent in sentences:
        if any(keyword in sent.lower() for keyword in important_keywords):
            if sent not in key_points and len(sent) > 20:  # Avoid duplicates and very short sentences
                key_points.append(sent)
    
    # If we still don't have enough points, add more sentences from the beginning
    if len(key_points) < 5:
        for sent in sentences[1:8]:  # Skip first sentence as we already added it
            if sent not in key_points and len(sent) > 20:
                key_points.append(sent)
    
    # Limit to 8 key points and ensure they're meaningful
    filtered_points = []
    for point in key_points[:8]:
        # Remove common boilerplate text
        if not any(phrase in point.lower() for phrase in ['equal opportunity', 'we are an equal', 
                                                          'please apply', 'send resume', 'contact us']):
            filtered_points.append(point)
    
    return filtered_points

# Define skill lists at module level so they're accessible throughout the app
tech_skills = [
    "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "php", "swift",
    "kotlin", "typescript", "html", "css", "sql", "nosql", "mongodb", "mysql", "postgresql",
    "oracle", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring",
    "asp.net", "jquery", "bootstrap", "tailwind", "aws", "azure", "gcp", "firebase",
    "docker", "kubernetes", "jenkins", "ci/cd", "git", "github", "bitbucket",
    "machine learning", "deep learning", "ai", "data science", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "data analysis", "data visualization", "power bi",
    "tableau", "excel", "vba", "sap", "salesforce", "jira", "confluence", "agile", "scrum"
]

soft_skills = [
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "time management", "project management", "analytical", "detail oriented", "creativity",
    "adaptability", "flexibility", "organization", "planning", "decision making",
    "conflict resolution", "presentation", "negotiation", "customer service", "interpersonal",
    "research", "writing", "editing", "public speaking", "mentoring", "coaching", "training"
]

# Then modify the extract_skills function to use these global lists
def extract_skills(text):
    """Extract skills from text using keyword matching"""
    # Use the global skill lists
    global tech_skills, soft_skills
    
    # Combine all skills
    all_skills = tech_skills + soft_skills
    
    # Find skills in the text
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        if skill in text_lower:
            found_skills.append(skill)
    
    return found_skills

# Ensure theme_mode is initialized before any usage
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

# Theme switcher for dark/light mode (top right, reliable)
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

# Place switcher at top right using columns
switcher_cols = st.columns([8,1])
with switcher_cols[1]:
    if st.button("🌙" if st.session_state["theme_mode"] == "dark" else "☀️", key="theme_switch_button", help="Toggle dark/light mode"):
        st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"

# Improved theme CSS for better contrast and visibility
if st.session_state["theme_mode"] == "dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #23272b !important; color: #e0e0e0 !important; }
        .stButton>button, .stDownloadButton button { background: #333 !important; color: #fff !important; border: 1px solid #888 !important; }
        .stTextArea textarea { background: #23272b !important; color: #e0e0e0 !important; border: 1px solid #888 !important; }
        .stExpander, .stExpanderHeader { background: #23272b !important; color: #e0e0e0 !important; border: 1px solid #888 !important; }
        .stTabs [data-baseweb="tab"] { background: #23272b !important; color: #e0e0e0 !important; }
        .stMarkdown, .markdown-text-container { color: #e0e0e0 !important; }
        .stMetric { color: #e0e0e0 !important; }
        .stAlert { background: #333 !important; color: #fff !important; }
        .stProgress > div > div { background: #b9ff39 !important; }
        .stProgress > div > div[role="progressbar"] { background: #23272b !important; }
        .stAlert[data-testid="stAlert-error"] { background: #ffcccc !important; color: #23272b !important; }
        .stAlert[data-testid="stAlert-warning"] { background: #fff3cd !important; color: #23272b !important; }
        .stAlert[data-testid="stAlert-success"] { background: #d4edda !important; color: #23272b !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
    body, .stApp { 
        background-color: #f5f5f5 !important; 
        color: #23272b !important; 
    }

    .stButton>button, .stDownloadButton button { 
        background: #fff !important; 
        color: #23272b !important; 
        border: 1px solid #bbb !important; 
    }

    .stTextArea textarea { 
        background: #fff !important; 
        color: #23272b !important; 
        border: 1px solid #bbb !important; 
    }

    .stExpander, .stExpanderHeader { 
        background: #fff !important; 
        color: #23272b !important; 
        border: 1px solid #bbb !important; 
    }
    .stExpander .stMarkdown, .stExpander .markdown-text-container {
        color: #23272b !important;
    }
    .stExpander .stCheckbox > div > label, .stExpander label, .stExpander .stForm label {
        color: #23272b !important;
    }
    .stForm .stMarkdown, .stForm .markdown-text-container {
        color: #23272b !important;
    }
    .stForm .stCheckbox > div > label, .stForm label {
        color: #23272b !important;
    }

    .stTabs [data-baseweb="tab"] { 
        background: #fff !important; 
        color: #23272b !important; 
    }

    .stMarkdown, .markdown-text-container { 
        color: #23272b !important; 
    }

    .stMetric { 
        color: #23272b !important; 
    }

    .stAlert { 
        background: #ffeaea !important; 
        color: #a94442 !important; 
        border: 1px solid #f5c2c7 !important; 
    }

    .stProgress > div > div { 
        background: #b9ff39 !important; 
    }

    .stProgress > div > div[role="progressbar"] { 
        background: #fff !important; 
    }

    .stAlert[data-testid="stAlert-error"] { 
        background: #ffcccc !important; 
        color: #a94442 !important; 
    }

    .stAlert[data-testid="stAlert-warning"] { 
        background: #fff3cd !important; 
        color: #23272b !important; 
    }

    .stAlert[data-testid="stAlert-success"] { 
        background: #d4edda !important; 
        color: #23272b !important; 
    }
</style>

    """, unsafe_allow_html=True)

# Main app function with Resume Enhancement section removed
def main():
    st.markdown("---")
    
    # File upload section
    st.subheader("Upload Your Files")
    col1, col2 = st.columns(2)
    
    with col1:
        resume_file = st.file_uploader("Choose your resume file", type=["pdf", "docx", "txt"])
    
    with col2:
        jd_file = st.file_uploader("Choose the job description file", type=["pdf", "docx", "txt"])
    
    # Process button and results - only show button when both files are present
    if resume_file is not None and jd_file is not None:
        # Show a full-width button
        analyze_button = st.button("Analyze Resume", use_container_width=True)
        
        # Only proceed with analysis when the button is clicked
        if analyze_button:
            with st.spinner("Processing your files..."):
                # Extract text from files
                resume_text = extract_text(resume_file)
                jd_text = extract_text(jd_file)
                
                # Calculate similarity score
                similarity_score = calculate_similarity_score(resume_text, jd_text)
                
                # Generate JD summary
                jd_summary = summarize_text(jd_text)
                
                # Extract key points from JD
                jd_key_points = extract_key_points(jd_text)
                
                # Extract skills from both resume and JD
                resume_skills = extract_skills(resume_text)
                jd_skills = extract_skills(jd_text)
                
                # Find missing skills
                missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
                
                # Store in session state for persistence
                st.session_state.resume_text = resume_text
                st.session_state.jd_text = jd_text
                st.session_state.similarity_score = similarity_score
                st.session_state.jd_summary = jd_summary
                st.session_state.jd_key_points = jd_key_points
                st.session_state.resume_skills = resume_skills
                st.session_state.jd_skills = jd_skills
                st.session_state.missing_skills = missing_skills
                
            # Display results
            st.subheader("Analysis Results")
            
            # Display match level message
            if similarity_score < 50:
                st.error("Your resume needs significant improvements to match this job description.")
            elif similarity_score < 70:
                st.warning("Your resume could use some targeted improvements for this job.")
            else:
                st.success("Your resume is well-matched to this job description!")
                
            # Additional visualization - Show key stats
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.metric(
                    label="Match Score", 
                    value=f"{similarity_score:.1f}%",
                    delta=f"{similarity_score - 50:.1f}%" if similarity_score > 50 else f"{similarity_score - 50:.1f}%",
                    delta_color="normal"
                )

            with stats_col2:
                # Calculate word count for resume
                resume_word_count = len(st.session_state.resume_text.split())
                st.metric(label="Resume Word Count", value=resume_word_count)

            with stats_col3:
                # Simple readability metric based on sentence length
                try:
                    sentences = st.session_state.resume_text.split('.')
                    words = st.session_state.resume_text.split()
                    avg_sentence_length = len(words) / max(len(sentences), 1)
                    # Simple heuristic: shorter sentences = better readability
                    readability_score = max(0, min(100, (25 - avg_sentence_length) * 4))
                    st.metric(label="Resume Readability", value=f"{readability_score:.1f}/100")
                except:
                    st.metric(label="Resume Readability", value="N/A")

            st.progress(similarity_score/100)
            
            # Job Description Summary expandable section
            with st.expander("Job Description Summary"):
                st.markdown("### Key Points from the Job Description")
                
                # Display key points as a bulleted list with improved formatting
                if st.session_state.jd_key_points:
                    for i, point in enumerate(st.session_state.jd_key_points, 1):
                        # Clean up the point text
                        clean_point = point.strip()
                        if clean_point:  # Only display non-empty points
                            # Remove common prefixes and clean up formatting
                            clean_point = clean_point.replace("• ", "").replace("- ", "")
                            # Ensure it doesn't end with unnecessary punctuation
                            if clean_point.endswith('.'):
                                clean_point = clean_point[:-1]
                            # Display with consistent bullet point formatting
                            st.markdown(f"**{i}.** {clean_point}")
                else:
                    st.info("No key points extracted from the job description.")
            
            # Missing Skills expandable section
            with st.expander("Skills Analysis"):
                # Create tabs for different views of skills analysis
                skill_tabs = st.tabs(["Missing Skills", "Resume Skills", "Skills Matching", "Resume Optimizer"])
                
                with skill_tabs[0]:  # Missing Skills tab
                    if st.session_state.missing_skills:
                        st.markdown("### Skills Missing in Your Resume")
                        st.markdown("These skills are mentioned in the job description but not found in your resume:")
                        
                        # Group missing skills by category
                        missing_tech_skills = [skill for skill in st.session_state.missing_skills 
                                            if skill in tech_skills]
                        missing_soft_skills = [skill for skill in st.session_state.missing_skills 
                                            if skill in soft_skills]
                        
                        # Create columns for better layout
                        missing_col1, missing_col2 = st.columns([1, 1])
                        
                        # Display missing skills by category in columns
                        with missing_col1:
                            if missing_tech_skills:
                                st.markdown("#### Technical Skills")
                                # Improved styling with better contrast
                                for skill in missing_tech_skills:
                                    st.markdown(f"""
                                    <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f0fa; border-radius: 5px; border: 1px solid #c0d6e9;'>
                                        <span style='font-size: 16px; color: #0a2540;'>⚙️ <b>{skill}</b></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        with missing_col2:
                            if missing_soft_skills:
                                st.markdown("#### Soft Skills")
                                # Improved styling with better contrast
                                for skill in missing_soft_skills:
                                    st.markdown(f"""
                                    <div style='padding: 8px 12px; margin: 5px 0; background-color: #f5f0fa; border-radius: 5px; border: 1px solid #d8c0e9;'>
                                        <span style='font-size: 16px; color: #0a2540;'>👥 <b>{skill}</b></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Priority score for missing skills (based on frequency in JD)
                        st.markdown("### Priority Skills to Add")
                        st.markdown("These missing skills appear most frequently in the job description:")
                        
                        # Count occurrences of each missing skill in the JD
                        skill_priorities = {}
                        jd_lower = st.session_state.jd_text.lower()
                        
                        for skill in st.session_state.missing_skills:
                            count = jd_lower.count(skill)
                            skill_priorities[skill] = count
                        
                        # Sort by count (descending)
                        sorted_priorities = sorted(skill_priorities.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)
                        
                        # Display top 5 priority skills with improved styling
                        priority_cols = st.columns(5)
                        for i, (skill, count) in enumerate(sorted_priorities[:5]):
                            with priority_cols[i]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 12px; background-color: #f0f7ff; border-radius: 5px; height: 100%; border: 1px solid #c0d6e9;'>
                                    <div style='font-size: 18px; font-weight: bold; color: #0a2540;'>{skill}</div>
                                    <div style='font-size: 14px; color: #486581;'>Mentioned {count} times</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Updated Skill Improvement Suggestions section with better text contrast

                        st.markdown("### Skill Improvement Suggestions")
                        st.markdown("To improve your resume, try to incorporate these missing skills in relevant sections:")

                        # Example suggestions with better formatting and improved text contrast
                        st.markdown("""
                        <div style='padding: 15px; background-color: #f8f9fa; border-left: 5px solid #4CAF50; margin-bottom: 10px;'>
                            <p style='color: #333333; margin: 5px 0;'><b>1. Add missing skills</b> to your <b>Skills</b> section</p>
                            <p style='color: #333333; margin: 5px 0;'><b>2. Incorporate these skills</b> in your <b>Work Experience</b> descriptions</p>
                            <p style='color: #333333; margin: 5px 0;'><b>3. Mention relevant skills</b> in your <b>Summary</b> or <b>Objective</b> statement</p>
                            <p style='color: #333333; margin: 5px 0;'><b>4. Highlight projects</b> demonstrating these skills in a <b>Projects</b> section</p>
                            <p style='color: #333333; margin: 5px 0;'><b>5. Consider online courses</b> to develop these skills further</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("Great job! Your resume already contains all the key skills mentioned in the job description.")
                
                with skill_tabs[1]:  # Resume Skills tab
                    st.markdown("### Skills Found in Your Resume")
                    
                    # Group resume skills by category
                    resume_tech_skills = [skill for skill in st.session_state.resume_skills 
                                         if skill in tech_skills]
                    resume_soft_skills = [skill for skill in st.session_state.resume_skills 
                                         if skill in soft_skills]
                    
                    # Calculate statistics
                    total_skills = len(st.session_state.resume_skills)
                    tech_percent = len(resume_tech_skills) / total_skills * 100 if total_skills > 0 else 0
                    soft_percent = len(resume_soft_skills) / total_skills * 100 if total_skills > 0 else 0
                    
                    # Display skill distribution chart
                    st.markdown("#### Skill Distribution")
                    
                    # Create consistent height and style for the chart
                    st.markdown("""
                    <style>
                    .stChart > div > div > svg {
                        height: 300px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    skill_chart_data = pd.DataFrame({
                        'Category': ['Technical Skills', 'Soft Skills'],
                        'Percentage': [tech_percent, soft_percent]
                    })
                    st.bar_chart(skill_chart_data.set_index('Category'))
                    
                    # Create container for skills display
                    skills_container = st.container()
                    
                    with skills_container:
                        # Display skills in a more organized grid with better styling
                        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                        
                        if resume_tech_skills:
                            st.markdown("#### Technical Skills")
                            # Create a grid layout
                            cols = st.columns(3)
                            for i, skill in enumerate(resume_tech_skills):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    if skill in st.session_state.jd_skills:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f4ea; border-radius: 5px; border: 1px solid #b7dfb9;'>
                                            <span style='font-size: 16px; color: #0a2540;'>✅ <b>{skill}</b></span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px; border: 1px solid #d0d7e2;'>
                                            <span style='font-size: 16px; color: #0a2540;'>ℹ️ {skill}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                        
                        if resume_soft_skills:
                            st.markdown("#### Soft Skills")
                            # Create a grid layout
                            cols = st.columns(3)
                            for i, skill in enumerate(resume_soft_skills):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    if skill in st.session_state.jd_skills:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #e6f4ea; border-radius: 5px; border: 1px solid #b7dfb9;'>
                                            <span style='font-size: 16px; color: #0a2540;'>✅ <b>{skill}</b></span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style='padding: 8px 12px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px; border: 1px solid #d0d7e2;'>
                                            <span style='font-size: 16px; color: #0a2540;'>ℹ️ {skill}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                
                with skill_tabs[2]:  # Skills Matching tab
                    st.markdown("### Skills Match Analysis")
                    
                    # Calculate matching statistics
                    total_jd_skills = len(st.session_state.jd_skills)
                    matching_skills = [skill for skill in st.session_state.resume_skills 
                                      if skill in st.session_state.jd_skills]
                    num_matching = len(matching_skills)
                    
                    # Create match percentage
                    match_percent = num_matching / total_jd_skills * 100 if total_jd_skills > 0 else 0
                    
                    # Display match statistics
                    st.metric(
                        label="Skills Match Rate", 
                        value=f"{match_percent:.1f}%",
                        help="Percentage of job description skills found in your resume"
                    )
                    
                    st.markdown(f"Your resume contains **{num_matching}** out of **{total_jd_skills}** skills mentioned in the job description.")
                    
                    # Display skill match visualization
                    match_data = {
                        'Status': ['Matching Skills', 'Missing Skills'],
                        'Count': [num_matching, total_jd_skills - num_matching]
                    }
                    match_df = pd.DataFrame(match_data)
                    
                    # Use a pie chart to visualize matching vs missing skills
                    fig = go.Figure(data=[go.Pie(
                        labels=match_df['Status'],
                        values=match_df['Count'],
                        hole=.3,
                        marker_colors=['#66BB6A', '#EF5350']
                    )])
                    fig.update_layout(title_text="Skills Match Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display matching skills
                    if matching_skills:
                        st.markdown("#### Matching Skills")
                        st.markdown("These skills from your resume match the job description requirements:")
                        
                        match_cols = st.columns(3)
                        for i, skill in enumerate(matching_skills):
                            col_idx = i % 3
                            with match_cols[col_idx]:
                                st.markdown(f"✅ {skill}")
                
                with skill_tabs[3]:  # Resume Optimizer tab
                    display_optimizer_tab(st.session_state)
            

            # Add Suggestion Optimizer section after Skills Analysis
            with st.expander("Suggestion Optimizer"):
                st.markdown("### Suggestions to Optimize Your Resume")
                skill_suggestions = [
                    "Add missing skills to your Skills section",
                    "Incorporate these skills in your Work Experience descriptions",
                    "Mention relevant skills in your Summary or Objective statement",
                    "Highlight projects demonstrating these skills in a Projects section",
                    "Consider online courses to develop these skills further"
                ]
                missing_skills = getattr(st.session_state, "missing_skills", [])
                all_suggestions = skill_suggestions + [f"Add skill: {skill}" for skill in missing_skills]

                def fallback_optimized_resume(resume_text, selected_suggestions):
                    return resume_text + "\n\nApplied Suggestions:\n" + "\n".join(selected_suggestions)

                with st.form("suggestion_optimizer_form"):
                    selected_suggestions = []
                    for suggestion in all_suggestions:
                        if st.checkbox(suggestion, key=f"sugg_{suggestion}"):
                            selected_suggestions.append(suggestion)
                    st.session_state.selected_suggestions = selected_suggestions
                    submitted = st.form_submit_button("Preview Optimized Resume")
                    if submitted:
                        resume_text = getattr(st.session_state, "resume_text", "")
                        jd_text = getattr(st.session_state, "jd_text", "")
                        if resume_text and jd_text and selected_suggestions:
                            try:
                                result = optimize_resume_with_openai(resume_text, jd_text, section_type="experience")
                                if result and result.strip():
                                    st.session_state.optimized_resume = result
                                else:
                                    st.session_state.optimized_resume = fallback_optimized_resume(resume_text, selected_suggestions)
                            except Exception:
                                st.session_state.optimized_resume = fallback_optimized_resume(resume_text, selected_suggestions)
                            st.session_state.optimized_resume_ready = True
                        else:
                            st.session_state.optimized_resume = fallback_optimized_resume(resume_text, selected_suggestions)
                            st.session_state.optimized_resume_ready = True

            # Always show preview after resume is generated (outside the form)
            if st.session_state.get("optimized_resume_ready", False):
                st.success("Preview of Optimized Resume below.")
                st.markdown("#### Optimized Resume Preview (Word)")
                word_buffer = create_word_doc(st.session_state.optimized_resume)
                st.download_button("Download as Word", data=word_buffer, file_name="optimized_resume.docx")
                st.markdown("---")
                st.markdown("##### Quick Text Preview")
                st.text_area("Optimized Resume Preview", value=st.session_state.optimized_resume, height=300)
            

def display_optimizer_tab(st_session_state):
    """Advanced Resume Optimizer with OpenAI"""
    st.markdown("### AI-Powered Resume Optimizer")
    st.markdown("Enhance your resume content using OpenAI's advanced language models while maintaining honesty and authenticity.")
    
    # Check if we have the required data first
    if not hasattr(st_session_state, 'resume_text') or not st_session_state.resume_text:
        st.warning("📋 No resume data found. Please upload and analyze your resume first.")
        st.info("👆 Go to the main page, upload both your resume and job description, then click 'Analyze Resume' to enable the optimizer features.")
        return
    
    if not hasattr(st_session_state, 'jd_text') or not st_session_state.jd_text:
        st.warning("📄 No job description data found. Please upload and analyze your job description first.")
        st.info("👆 Go to the main page, upload both your resume and job description, then click 'Analyze Resume' to enable the optimizer features.")
        return
    
    # Check if API key is configured
    if not openai_client:
        st.error("🔑 OpenAI API key not configured. Please add your OPENAI_API_KEY to the .env file or ensure the API key is properly set.")
        st.info("💡 The application will fall back to basic suggestions if the API is not available.")
    
    # Show data status
    st.success(f"✅ Data loaded: Resume ({len(st_session_state.resume_text)} chars) | Job Description ({len(st_session_state.jd_text)} chars)")
    
    # Create optimization tabs
    opt_tabs = st.tabs(["Role Relevance Matrix (AI Inference)", "Keyword Enhancement", "ATS Optimization"])

    with opt_tabs[0]:  # Role Relevance Matrix (AI Inference)
        st.markdown("#### Role Relevance Matrix (AI Inference)")
        st.markdown("""
        <div style='color:#b9f6ca; font-size:18px; margin-bottom:10px;'>
        <b>Based on your resume, you're best suited for roles like:</b>
        </div>
        """, unsafe_allow_html=True)
        # Deterministic role inference based on resume keywords/sections
        resume_text = st_session_state.resume_text.lower()
        inferred_roles = []
        # Simple keyword-based logic for demonstration
        if any(word in resume_text for word in ["cloud", "aws", "azure", "gcp", "architecture", "infrastructure"]):
            inferred_roles.append("Solutions Architect – Cloud AI")
        if any(word in resume_text for word in ["product", "roadmap", "stakeholder", "requirements", "agile", "scrum"]):
            inferred_roles.append("Technical Product Manager")
        if any(word in resume_text for word in ["customer", "success", "support", "b2b", "saas", "onboarding"]):
            inferred_roles.append("Customer Success Engineer in B2B SaaS")
        if any(word in resume_text for word in ["data", "ml", "ai", "analytics", "python", "model", "deep learning"]):
            inferred_roles.append("Machine Learning Engineer")
        if any(word in resume_text for word in ["frontend", "react", "vue", "ui", "javascript", "css"]):
            inferred_roles.append("Frontend Engineer / UI Developer")
        if any(word in resume_text for word in ["backend", "api", "database", "django", "node.js", "sql"]):
            inferred_roles.append("Backend Engineer / API Developer")
        if not inferred_roles:
            inferred_roles = ["Generalist Software Engineer", "Technical Consultant", "IT Analyst"]
        # Show the roles
        for role in inferred_roles:
            st.markdown(f"<div style='padding:12px; background:#23272b; color:#fff; border-radius:6px; margin-bottom:8px; font-size:17px;'><b>{role}</b></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-top:18px; color:#bdbdbd;'>
        <i>Though you're applying for your selected role, these adjacent roles could offer better alignment based on your experience and skills.</i>
        </div>
        """, unsafe_allow_html=True)
    with opt_tabs[1]:  # Keyword Enhancement
        st.markdown("#### Strategic Keyword Enhancement")
        
        # Check if we have required data
        if not hasattr(st_session_state, 'resume_text') or not hasattr(st_session_state, 'jd_text'):
            st.warning("Please analyze your resume and job description first.")
            st.info("👆 Go to the main page, upload both your resume and job description, then click 'Analyze Resume' to enable keyword enhancement.")
            return
        
        try:
            # Find missing keywords with context
            missing_keywords = find_contextual_keywords(
                st_session_state.resume_text,
                st_session_state.jd_text
            )
            
            if not missing_keywords:
                st.success("✅ Great! Your resume already contains most keywords from the job description.")
                st.info("Consider reviewing the other tabs for additional optimization suggestions.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### High-Impact Keywords to Add")
                
                for keyword, context in missing_keywords[:5]:
                    st.markdown(f"""
                    <div style='padding: 10px; background-color: #f0f7ff; border-radius: 5px; margin: 5px 0; border: 1px solid #c0d6e9;'>
                        <b style='color: #0a2540;'>{keyword}</b><br>
                        <small style='color: #666;'>Context: {context}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("##### Suggested Placement")
                
                # Extract resume sections before suggesting keyword placement
                resume_sections = extract_resume_sections(st_session_state.resume_text)
                # Suggest where to place keywords
                placement_suggestions = suggest_keyword_placement(missing_keywords, resume_sections)
                
                if placement_suggestions:
                    # Replace nested expander with simple markdown containers
                    for section, keywords in placement_suggestions.items():
                        st.markdown(f"**Add to {section.title()} Section:**")
                        for keyword in keywords:
                            st.markdown(f"• {keyword}")
                        st.markdown("---")
                else:
                    st.info("Consider adding these keywords to relevant sections of your resume.")
                    
        except Exception as e:
            st.error(f"❌ Error analyzing keywords: {str(e)}")
            st.info("Please try uploading your files again or contact support if the issue persists.")
    
    with opt_tabs[2]:  # ATS Optimization
        st.markdown("#### ATS Compatibility Check")
        
        # Check if we have required data
        if not hasattr(st_session_state, 'resume_text') or not st_session_state.resume_text:
            st.warning("Please analyze your resume first.")
            st.info("👆 Go to the main page, upload your resume, and click 'Analyze Resume' to enable ATS optimization.")
            return
        
        try:
            # Run ATS compatibility analysis (stricter version)
            ats_score, ats_issues = analyze_ats_compatibility_strict(st_session_state.resume_text, st_session_state.jd_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ATS Compatibility Score", f"{ats_score:.1f}%")
                st.progress(ats_score / 100)
                if ats_score >= 90:
                    st.success("✅ Excellent ATS compatibility!")
                elif ats_score >= 70:
                    st.warning("⚠️ Good compatibility with room for improvement")
                else:
                    st.error("❌ Poor ATS compatibility - needs significant improvement")
            
            with col2:
                st.markdown("##### ATS Issues Found")
                if ats_issues:
                    for issue in ats_issues:
                        st.markdown(f"<div style='background-color:#2d2d2d; color:#fff; padding:10px; border-radius:5px; margin-bottom:5px;'>⚠️ {issue}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color:#1b5e20; color:#fff; padding:10px; border-radius:5px;'>✅ No major ATS issues detected!</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("##### ATS Optimization Recommendations")
            ats_recommendations = generate_ats_recommendations(ats_issues, st_session_state.resume_text)
            if ats_recommendations:
                for i, rec in enumerate(ats_recommendations, 1):
                    st.markdown(f"""
                    <div style='padding: 12px; background-color: #23272b; color: #fff; border-left: 4px solid #4CAF50; margin: 10px 0; border-radius: 5px;'>
                        <b style='color:#b9f6ca;'>{i}. {rec['title']}</b><br>
                        <span style='color: #fff;'>{rec['description']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific recommendations available. Your resume structure looks good!")
        except Exception as e:
            st.error(f"❌ Error analyzing ATS compatibility: {str(e)}")
            st.info("Please try uploading your resume again or contact support if the issue persists.")

def analyze_ats_compatibility_strict(resume_text, jd_text):
    """Stricter ATS compatibility analysis"""
    issues = []
    score = 100
    # Section checks
    required_sections = ["experience", "education", "skills", "summary"]
    for sec in required_sections:
        if not re.search(sec, resume_text, re.IGNORECASE):
            issues.append(f"Missing '{sec.title()}' section.")
            score -= 15
    # Word count
    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append("Resume may be too short for proper ATS parsing (min 200 words recommended).")
        score -= 20
    elif word_count > 1200:
        issues.append("Resume may be too long for ATS (max 1200 words recommended).")
        score -= 10
    # Formatting
    if len(re.findall(r'[^\w\s]', resume_text)) > len(resume_text) * 0.08:
        issues.append("Too many special characters may confuse ATS.")
        score -= 10
    # Bullet points
    if resume_text.count('\n-') + resume_text.count('\n•') < 3:
        issues.append("Not enough bullet points detected. Use bullets for clarity.")
        score -= 10
    # Keyword match (at least 50% of JD keywords should be present)
    jd_keywords = set([w.lower() for w in jd_text.split() if len(w) > 4])
    resume_words = set([w.lower() for w in resume_text.split() if len(w) > 4])
    match_count = len(jd_keywords & resume_words)
    if len(jd_keywords) > 0 and match_count / len(jd_keywords) < 0.5:
        issues.append("Less than 50% of job description keywords found in resume.")
        score -= 15
    # Cap score between 0 and 99
    score = max(0, min(score, 99))
    return score, issues

# Helper functions for the optimizer

def extract_resume_sections(resume_text):
    """Extract different sections from resume using regex and NLP"""
    sections = {}
    
    # Common section headers
    section_patterns = {
        'experience': r'(?i)(experience|employment|work history|professional experience)(.*?)(?=(?:education|skills|projects|certifications|$))',
        'skills': r'(?i)(skills|technical skills|competencies)(.*?)(?=(?:experience|education|projects|certifications|$))',
        'education': r'(?i)(education|academic)(.*?)(?=(?:experience|skills|projects|certifications|$))',
        'summary': r'(?i)(summary|objective|profile)(.*?)(?=(?:experience|education|skills|projects|$))'
    }
    
    for section, pattern in section_patterns.items():
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            sections[section] = match.group(2).strip()
    
    return sections

def extract_jd_requirements(jd_text):
    """Extract key requirements from job description"""
    doc = nlp(jd_text)
    
    requirements = []
    
    # Look for requirement indicators
    requirement_indicators = ['required', 'must have', 'should have', 'preferred', 'experience with']
    
    for sent in doc.sents:
        if any(indicator in sent.text.lower() for indicator in requirement_indicators):
            requirements.append(sent.text.strip())
    
    return requirements

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if 'semantic_model' not in st.session_state:
        return 0
    
    try:
        embeddings = st.session_state.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity * 100
    except:
        return 0

def find_matching_bullets(experience_text, jd_requirements, threshold=0.3):
    """Find bullet points that match JD requirements"""
    bullets = extract_bullet_points(experience_text)
    matching_bullets = []
    
    for bullet in bullets:
        for req in jd_requirements:
            similarity = calculate_semantic_similarity(bullet, req) / 100
            if similarity > threshold:
                matching_bullets.append(bullet)
                break
    
    return matching_bullets

def extract_bullet_points(text):
    """Extract bullet points from text"""
    lines = text.split('\n')
    bullets = []
    
    for line in lines:
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
            bullets.append(line.lstrip('•-* '))
        elif re.match(r'^\d+\.', line):  # Numbered lists
            bullets.append(re.sub(r'^\d+\.\s*', '', line))
    
    return bullets

def calculate_bullet_relevance(bullet, jd_requirements):
    """Calculate how relevant a bullet point is to JD requirements"""
    if not jd_requirements:
        return 0
    
    max_similarity = 0
    for req in jd_requirements:
        similarity = calculate_semantic_similarity(bullet, req) / 100
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def optimize_bullet_point(bullet, jd_requirements):
    """Optimize a bullet point for better JD alignment while maintaining honesty"""
    # This is a simplified version - in production, you'd use T5/BART for paraphrasing
    
    # Extract action verbs and achievements
    action_verbs = ['developed', 'implemented', 'managed', 'led', 'created', 'improved', 'analyzed', 'designed']
    
    # Find the most relevant requirement
    best_req = ""
    best_similarity = 0
    
    for req in jd_requirements:
        similarity = calculate_semantic_similarity(bullet, req) / 100
        if similarity > best_similarity:
            best_similarity = similarity
            best_req = req
    
    # Simple optimization: enhance with stronger action verbs and quantifiers
    optimized = bullet
    
    # Replace weak verbs with stronger ones
    weak_to_strong = {
        'worked on': 'developed',
        'helped': 'contributed to',
        'did': 'executed',
        'made': 'created'
    }
    
    for weak, strong in weak_to_strong.items():
        if weak in optimized.lower():
            optimized = optimized.replace(weak, strong)
    
    # Add quantifiers if missing
    if not re.search(r'\d+', optimized):
        optimized = optimized + " (quantify with specific metrics when possible)"
    
    return optimized

def highlight_changes(original, optimized):
    """Highlight changes between original and optimized text"""
    # Simple diff highlighting
    if original == optimized:
        return "No changes made"
    
    return f"""
    <div style='background-color: #ffe6e6; padding: 5px; margin: 5px 0;'>
        <del>{original}</del>
    </div>
    <div style='background-color: #e6ffe6; padding: 5px; margin: 5px 0;'>
        <ins>{optimized}</ins>
    </div>
    """

def identify_weak_sections(resume_sections, jd_requirements):
    """Identify sections that need improvement"""
    weak_sections = {}
    
    for section, content in resume_sections.items():
        if content:
            score = calculate_semantic_similarity(content, ' '.join(jd_requirements))
            if score < 30:  # Low alignment threshold
                weak_sections[section] = [
                    "Add more relevant keywords from the job description",
                    "Include specific achievements and metrics",
                    "Align language with job requirements"
                ]
    
    return weak_sections

def find_contextual_keywords(resume_text, jd_text):
    """Find missing keywords with their context"""
    try:
        # Extract important terms from JD
        jd_doc = nlp(jd_text)
        resume_doc = nlp(resume_text)
        
        # Get important terms (nouns, adjectives that aren't in resume)
        jd_terms = set()
        resume_terms = set()
        
        for token in jd_doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2 and not token.is_stop:
                jd_terms.add(token.lemma_.lower())
        
        for token in resume_doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2 and not token.is_stop:
                resume_terms.add(token.lemma_.lower())
        
        missing_terms = jd_terms - resume_terms
        
        # Get context for missing terms
        contextual_keywords = []
        for term in list(missing_terms)[:10]:  # Limit to top 10
            # Find sentence containing the term in JD
            for sent in jd_doc.sents:
                if term in sent.text.lower():
                    context = sent.text[:100] + "..." if len(sent.text) > 100 else sent.text
                    contextual_keywords.append((term, context))
                    break
        
        return contextual_keywords
    
    except Exception as e:
        st.warning(f"Error analyzing keywords: {str(e)}")
        return []

def suggest_keyword_placement(missing_keywords, resume_sections):
    """Suggest where to place missing keywords"""
    placement = {}
    
    for keyword, context in missing_keywords:
        # Simple heuristic for placement
        if any(tech in keyword.lower() for tech in ['python', 'java', 'sql', 'aws']):
            if 'skills' not in placement:
                placement['skills'] = []
            placement['skills'].append(keyword)
        else:
            if 'experience' not in placement:
                placement['experience'] = []
            placement['experience'].append(keyword)
    
    return placement

def analyze_ats_compatibility(resume_text):
    """Analyze ATS compatibility issues"""
    issues = []
    score = 100
    
    # Check for common ATS issues
    if len(resume_text.split()) < 200:
        issues.append("Resume may be too short for proper ATS parsing")
        score -= 15
    
    if not re.search(r'(skills?|technical|competencies)', resume_text, re.IGNORECASE):
        issues.append("No clear skills section detected")
        score -= 20
    
    if not re.search(r'(experience|employment|work)', resume_text, re.IGNORECASE):
        issues.append("No clear experience section detected")
        score -= 25
    
    # Check for formatting issues
    if len(re.findall(r'[^\w\s]', resume_text)) > len(resume_text) * 0.1:
        issues.append("Too many special characters may confuse ATS")
        score -= 10
    
    return max(0, score), issues

def generate_ats_recommendations(issues, resume_text):
    """Generate ATS optimization recommendations"""
    recommendations = []
    
    if "too short" in str(issues):
        recommendations.append({
            'title': 'Expand Content',
            'description': 'Add more detailed descriptions of your achievements and responsibilities'
        })
    
    if "skills section" in str(issues):
        recommendations.append({
            'title': 'Add Skills Section',
            'description': 'Create a dedicated "Skills" or "Technical Skills" section with relevant keywords'
        })
    
    if "experience section" in str(issues):
        recommendations.append({
            'title': 'Organize Experience',
            'description': 'Create a clear "Professional Experience" or "Work Experience" section'
        })
    
    # Always include these best practices
    recommendations.extend([
        {
            'title': 'Use Standard Headers',
            'description': 'Use conventional section headers like "Experience", "Education", "Skills"'
        },
        {
            'title': 'Include Keywords',
            'description': 'Incorporate exact keywords from the job description naturally in your content'
        }
    ])
    
    return recommendations

def optimize_resume_with_openai(resume_text, jd_text, section_type="experience"):
    """Use OpenAI to optimize resume content"""
    try:
        # Check if the API client is available
        if not openai_client:
            st.warning("⚠️ OpenAI API not available, using basic optimization...")
            return f"""**Basic Optimization Suggestions for {section_type.title()} Section:**

• Use stronger action verbs (led, implemented, developed, managed)
• Add quantifiable metrics and achievements
• Include relevant keywords from the job description
• Format as clear bullet points
• Focus on results and impact

**Your original content with suggested improvements:**
{resume_text}

*For AI-powered optimization, please ensure your OpenAI API key is properly configured.*"""
        
        # Create optimization prompt
        prompt = f"""As a professional resume writer, help optimize this resume section for the given job description.

**Job Description:**
{jd_text[:1200]}

**Current Resume Section ({section_type}):**
{resume_text}

**Instructions:**
1. Rewrite the resume section to better align with the job description
2. Use strong action verbs and quantifiable achievements
3. Include relevant keywords from the job description naturally
4. Maintain honesty - don't add false information
5. Keep the same experience and skills, just improve the presentation
6. Format as bullet points where appropriate

**Optimized Resume Section:**"""
        
        # Generate optimized content
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional resume writer with expertise in optimizing resume content for specific job descriptions while maintaining authenticity and honesty."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1200,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            else:
                return "Error: Empty response received from OpenAI API."
        else:
            return "Error: No response received from OpenAI API."
        
    except Exception as e:
        error_msg = f"Error optimizing with OpenAI: {str(e)}"
        st.warning(f"⚠️ {error_msg}")
        st.info("🔄 Providing basic optimization suggestions instead...")
        
        return f"""**Basic Optimization Suggestions for {section_type.title()} Section:**

• Use stronger action verbs (led, implemented, developed, managed)
• Add quantifiable metrics and achievements  
• Include relevant keywords from the job description
• Format as clear bullet points
• Focus on results and impact

**Your original content:**
{resume_text}

**Error details:** {str(e)}
*Please check your internet connection and API key configuration.*"""

def generate_resume_suggestions_with_openai(resume_text, jd_text):
    """Generate improvement suggestions using OpenAI"""
    try:
        # Check if the API client is available
        if not openai_client:
            st.warning("⚠️ OpenAI API client not configured.")
            return generate_fallback_suggestions(resume_text, jd_text)
        
        # Check if we have valid input
        if not resume_text or not jd_text:
            st.error("❌ Both resume text and job description are required.")
            return "Error: Missing resume or job description text."
        
        # Truncate inputs to prevent token limit issues
        resume_truncated = resume_text[:1800] if len(resume_text) > 1800 else resume_text
        jd_truncated = jd_text[:1200] if len(jd_text) > 1200 else jd_text
        
        st.info("🚀 Starting OpenAI API call...")
        st.info(f"📝 Resume length: {len(resume_truncated)} characters")
        st.info(f"📋 Job description length: {len(jd_truncated)} characters")
        
        # Create the prompt
        prompt = f"""As an expert resume reviewer, analyze this resume against the job description and provide specific improvement suggestions.

**Job Description:**
{jd_truncated}

**Resume:**
{resume_truncated}

**Please provide:**
1. Top 5 specific improvements needed
2. Missing keywords that should be added
3. Suggested action verbs to use
4. Recommendations for quantifying achievements
5. Overall strategy for better alignment

Format your response with clear sections and bullet points. Be specific and actionable."""
        
        # Make the API call
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert resume reviewer and career coach. Provide specific, actionable advice for improving resumes based on job descriptions."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        st.success("✅ OpenAI API call completed!")
        
        # Extract the response
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                st.success("🎉 Successfully generated AI suggestions!")
                return content.strip()
            else:
                st.warning("⚠️ Empty response received from OpenAI.")
                return generate_fallback_suggestions(resume_text, jd_text)
        else:
            st.error("❌ No valid response received from OpenAI.")
            return generate_fallback_suggestions(resume_text, jd_text)
        
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {str(e)}")
        st.info("🔄 Falling back to basic suggestions...")
        return generate_fallback_suggestions(resume_text, jd_text)

def generate_fallback_suggestions(resume_text, jd_text):
    """Generate basic suggestions without AI when OpenAI API is not available"""
    suggestions = """
## Resume Improvement Suggestions (Basic Analysis)

### 1. Top 5 Specific Improvements Needed
• **Add quantifiable achievements**: Include specific numbers, percentages, and metrics in your experience descriptions
• **Use stronger action verbs**: Replace weak verbs like "helped" with powerful verbs like "spearheaded", "optimized", "implemented"
• **Include relevant keywords**: Add industry-specific terms and skills mentioned in the job description
• **Improve formatting**: Ensure consistent formatting with clear section headers and bullet points
• **Tailor content**: Customize your resume content to better match the specific job requirements

### 2. Missing Keywords Analysis
Based on the job description, consider adding these types of keywords to your resume:
• Technical skills and tools mentioned in the job posting
• Industry-specific terminology and buzzwords
• Required certifications or qualifications
• Soft skills emphasized in the job description
• Company values and culture-related terms

### 3. Suggested Action Verbs
Consider using these powerful action verbs in your resume:
• **Leadership**: Led, Directed, Managed, Supervised, Coordinated
• **Achievement**: Achieved, Accomplished, Delivered, Exceeded, Improved
• **Innovation**: Developed, Created, Designed, Implemented, Launched
• **Analysis**: Analyzed, Evaluated, Researched, Investigated, Assessed
• **Communication**: Presented, Communicated, Collaborated, Facilitated, Negotiated

### 4. Recommendations for Quantifying Achievements
• Replace vague statements with specific metrics
• Include percentage improvements (e.g., "Increased sales by 25%")
• Add dollar amounts for cost savings or revenue generation
• Mention team sizes you managed or worked with
• Include project timelines and deadlines met

### 5. Overall Strategy for Better Alignment
• **Mirror the job description language**: Use similar terminology and phrasing
• **Highlight relevant experience**: Prioritize experiences that match job requirements
• **Address gaps**: Acknowledge and address any missing qualifications
• **Showcase transferable skills**: Emphasize skills that apply across industries
• **Optimize for ATS**: Use standard section headers and include keywords naturally

### Note
For more personalized and detailed suggestions, please configure your OpenAI API key in the .env file to enable AI-powered analysis.
"""
    return suggestions

main()