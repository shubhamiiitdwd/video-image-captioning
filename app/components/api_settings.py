import streamlit as st
import requests
from ..utils.config import get_huggingface_api_key

def check_hf_api_status():
    """Check if Hugging Face API is currently available"""
    try:
        # Make a simple request to check API status
        response = requests.get("https://huggingface.co/api/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def render_api_settings():
    """Render API settings component"""
    st.markdown("### 🔑 API Settings")
    
    # Check API status 
    api_available = check_hf_api_status()
    
    # Initialize session state for API key if not present
    if "huggingface_api_key" not in st.session_state:
        st.session_state.huggingface_api_key = get_huggingface_api_key()
    
    # Display API status
    if api_available:
        st.success("✅ Hugging Face API services are online")
    else:
        st.error("⚠️ Hugging Face API services appear to be offline or experiencing issues")
        st.info("The application will use local models for processing. This may be slower but will work without an API connection.")
    
    # Display API key input
    with st.expander("Hugging Face API Key Settings"):
        st.markdown("""
        You need a Hugging Face API key to use the hosted models. If you don't provide a key,
        the application will download and use models locally, which might be slower and requires more memory.
        
        To get an API key:
        1. Sign up at [Hugging Face](https://huggingface.co/join)
        2. Go to your [profile settings](https://huggingface.co/settings/tokens)
        3. Create a new API token
        """)
        
        # API key input
        api_key = st.text_input(
            "Enter your Hugging Face API key",
            value=st.session_state.huggingface_api_key,
            type="password",
            help="Your API key will be stored in the session state.",
            key="api_key_input"
        )
        
        if st.button("Save API Key"):
            st.session_state.huggingface_api_key = api_key
            if api_key:
                st.success("API key saved to session state! Using this key for model inference.")
            else:
                st.warning("No API key provided. Using local models.")
        
        # Option to use local models
        use_local = st.checkbox(
            "Use local models instead of API",
            value=not bool(st.session_state.huggingface_api_key) or not api_available,
            help="Check this to download and use models locally instead of using the API",
            key="use_local_models"
        )
        
        if not api_available and not use_local:
            st.warning("API appears to be offline. Forcing local model usage.")
            use_local = True
        
        if use_local:
            st.info("Using local models. First run may download models which can take some time.")
    
    return st.session_state.huggingface_api_key, use_local 