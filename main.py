import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import os
from streamlit_extras.app_logo import add_logo
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import time
import json
import zipfile
import shutil
import uuid
from tasks import run_pockethunter_pipeline, run_extract_to_pdb_task, run_detect_pockets_task, run_cluster_pockets_task, run_docking_task
from celery_app import celery_app

# Page configuration
st.set_page_config(
    page_title="PocketHunter Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo/pockethunter',
        'Report a bug': "https://github.com/your-repo/pockethunter/issues",
        'About': "# PocketHunter Suite\nA modern molecular dynamics pocket detection and analysis tool."
    }
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .status-success {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-error {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-info {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .job-id-display {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="main-header">
    <h1>🧬 PocketHunter Suite</h1>
    <p>Advanced Molecular Dynamics Pocket Detection & Analysis</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for job ID caching
if 'cached_job_ids' not in st.session_state:
    st.session_state.cached_job_ids = {
        'extract': None,
        'detect': None,
        'cluster': None
    }

# Clear old session state that might cause issues
def clear_old_session_state():
    """Clear old session state that might cause Celery errors"""
    old_keys = [
        'current_pipeline_job_id', 'pipeline_task_id', 'pipeline_done',
        'extract_task_id', 'detect_task_id', 'cluster_task_id'
    ]
    for key in old_keys:
        if key in st.session_state:
            del st.session_state[key]

# Clear old session state
clear_old_session_state()

# Define the pages (removed Full Pipeline)
pages = {
    "Step 1: Extract Frames": "extract_frames_app.py", 
    "Step 2: Detect Pockets": "detect_pockets_app.py",
    "Step 3: Cluster Pockets": "cluster_pockets_app.py",
    "Step 4: Molecular Docking": "docking_app.py",
    "Task Monitor": "task_monitor_app.py"
}

# Horizontal menu
selected = option_menu(
    None, 
    list(pages.keys()), 
    icons=['file-earmark-arrow-down', 'search', 'diagram-3', 'flask', 'activity'], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#667eea", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#667eea"},
        "nav-link-selected": {"background-color": "#667eea"},
    }
)

# Get the corresponding file
selected_file = pages[selected]

# Execute the selected file
with open(selected_file) as f:
    code = f.read()
    exec(code) 