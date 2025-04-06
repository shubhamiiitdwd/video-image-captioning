import streamlit as st
import os
import asyncio
import nest_asyncio
import subprocess
import sys

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# Import components directly to avoid import resolution issues
from app.components.api_settings import render_api_settings
from app.components.image_captioning import render_image_captioning
from app.components.video_captioning import render_video_captioning

# Set page config
st.set_page_config(
    page_title="Media Captioning Hub",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize asyncio for Windows compatibility
def init_asyncio():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Get current event loop or create a new one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop

# Load custom CSS
def load_css():
    css = """
    <style>
        /* Main title styling */
        .main-title {
            color: #2C3E50;
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 2px solid #3498DB;
            margin-bottom: 30px;
        }
        
        /* Section headers */
        .section-header {
            color: #3498DB;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            margin: 20px 0 15px 0;
        }
        
        /* Caption container */
        .caption-container, .video-caption-container {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #3498DB;
        }
        
        /* Speech transcript styling */
        .speech-transcript {
            border-left: 4px solid #9B59B6;
        }
        
        /* Translated caption styling */
        .translated-caption {
            border-left: 4px solid #2ECC71;
        }
        
        /* Combined caption styling */
        .combined-caption {
            border-left: 4px solid #E74C3C;
        }
        
        /* Caption text styling */
        .caption-text {
            font-size: 18px;
            line-height: 1.5;
        }
        
        /* Timestamp styling */
        .timestamp {
            font-weight: bold;
            color: #7F8C8D;
            font-family: monospace;
        }
        
        /* General section styling */
        .section {
            background-color: rgba(52, 152, 219, 0.1);
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    # Initialize asyncio
    loop = init_asyncio()
    
    # Load CSS
    load_css()
    
    # App title and description
    st.markdown("<h1 class='main-title'>🎞️ Media Captioning Hub</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section">
    <p>Generate captions for images and videos using state-of-the-art AI models. 
    Extract speech from videos and get accurate transcriptions with timestamps. 
    Translate captions to multiple languages and export them to various formats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render API settings in sidebar
    with st.sidebar:
        st.image("https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png", width=200)
        api_key, use_local_models = render_api_settings()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📷 Image Captioning", "🎬 Video Captioning"])
    
    # Tab content
    with tab1:
        render_image_captioning(api_key, use_local_models)
    
    with tab2:
        render_video_captioning(api_key, use_local_models)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #ccc;">
        <p>Powered by <a href="https://huggingface.co" target="_blank">Hugging Face</a> models and <a href="https://streamlit.io" target="_blank">Streamlit</a>.</p>
    </div>
    """, unsafe_allow_html=True)

# Create temp folder if it doesn't exist
if not os.path.exists("temp"):
    try:
        os.makedirs("temp")
    except FileExistsError:
        # Directory already exists, which is fine
        pass

if __name__ == "__main__":
    main() 