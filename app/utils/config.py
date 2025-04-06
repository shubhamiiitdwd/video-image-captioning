import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Hugging Face API config
def get_huggingface_api_key():
    """
    Get Hugging Face API key from environment variables or Streamlit secrets
    """
    # Try to get API key from environment variables first
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    # If not found in env vars, check if stored in session state
    if not api_key or api_key == "your_huggingface_api_key_here":
        api_key = st.session_state.get("huggingface_api_key", "")
    
    return api_key

# Available models - Expanded with more options
IMAGE_CAPTION_MODELS = {
    # BLIP Models
    "Salesforce/blip-image-captioning-base": "BLIP Base",
    "Salesforce/blip-image-captioning-large": "BLIP Large",
    
    # BLIP-2 Models (more advanced)
    "Salesforce/blip2-opt-2.7b": "BLIP-2 OPT-2.7B (High accuracy)",
    "Salesforce/blip2-opt-6.7b": "BLIP-2 OPT-6.7B (Very high accuracy)",
    "Salesforce/blip2-flan-t5-xl": "BLIP-2 FLAN-T5-XL (Best quality)",
    
    # GIT Models
    "microsoft/git-base-coco": "GIT Base COCO",
    "microsoft/git-large-coco": "GIT Large COCO",
    
    # ViT-GPT2 Models
    "nlpconnect/vit-gpt2-image-captioning": "ViT-GPT2 (Fast)",
    
    # OFA Models
    "OFA-Sys/ofa-base": "OFA Base (Multi-task)",
    
    # CLIP+GPT Models
    "clip-gpt/clip-gpt": "CLIP+GPT (Creative captions)"
}

# Speech recognition models
SPEECH_RECOGNITION_MODELS = {
    "openai/whisper-tiny": "Whisper Tiny (Fast, less accurate)",
    "openai/whisper-small": "Whisper Small (Balanced)",
    "openai/whisper-base": "Whisper Base (More accurate, slower)"
}

# Translation models - Expanded
TRANSLATION_MODELS = {
    # Base models
    "t5-base": "T5 Base (English)",
    "facebook/m2m100_418M": "M2M100 (418M, Multi-language)",
    "facebook/m2m100_1.2B": "M2M100 (1.2B, High accuracy)",
    
    # Language-specific models
    "Helsinki-NLP/opus-mt-en-fr": "English to French",
    "Helsinki-NLP/opus-mt-en-es": "English to Spanish",
    "Helsinki-NLP/opus-mt-en-de": "English to German",
    "Helsinki-NLP/opus-mt-en-zh": "English to Chinese",
    "Helsinki-NLP/opus-mt-en-hi": "English to Hindi",
    "Helsinki-NLP/opus-mt-en-ar": "English to Arabic",
    "Helsinki-NLP/opus-mt-en-ru": "English to Russian",
    "Helsinki-NLP/opus-mt-en-ja": "English to Japanese",
    "Helsinki-NLP/opus-mt-en-it": "English to Italian",
    "Helsinki-NLP/opus-mt-en-pt": "English to Portuguese",
    "Helsinki-NLP/opus-mt-en-ko": "English to Korean"
}

# Supported languages for UI
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "hi": "Hindi",
    "ar": "Arabic",
    "ru": "Russian",
    "ja": "Japanese",
    "it": "Italian",
    "pt": "Portuguese",
    "ko": "Korean"
}

# Caption generation settings
CAPTION_STYLES = {
    "default": "Standard Caption",
    "detailed": "Detailed Description",
    "concise": "Short & Concise",
    "creative": "Creative & Expressive",
    "technical": "Technical Description"
}

# Caption prompts for different styles
CAPTION_PROMPTS = {
    "default": "",
    "detailed": "Provide a detailed description of this image, mentioning all visible elements and their characteristics:",
    "concise": "Describe this image in one brief sentence:",
    "creative": "Describe this image in a creative and expressive way, using vivid language:",
    "technical": "Provide a technical analysis of this image, mentioning objects, positions, colors, and any notable technical aspects:"
}

# Advanced settings
DEFAULT_ADVANCED_SETTINGS = {
    "beam_size": 5,
    "max_length": 50,
    "min_length": 5,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "num_frames_video": 20,
    "ensemble_models": False,
    "apply_post_processing": True,
    "enhanced_preprocessing": False
} 