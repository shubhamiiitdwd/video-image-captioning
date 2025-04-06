import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from .config import get_huggingface_api_key
import requests
import streamlit as st

class CaptionTranslator:
    def __init__(self, model_name="t5-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.api_key = get_huggingface_api_key()
    
    def load_model(self):
        """Load the translation model"""
        if self.model is None:
            if "t5" in self.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.model = self.model.to(self.device)
        return self.model, self.tokenizer
    
    def translate_text_api(self, text):
        """Translate text using Hugging Face API"""
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
        
        # API endpoint
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # API request payload
        payload = {"inputs": text}
        
        try:
            # API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                # If API is down (503), fall back to local model
                if response.status_code == 503:
                    st.warning("Hugging Face API is currently unavailable for translation. Falling back to local model.")
                    return self.translate_text_local(text)
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # Parse and return the result
            result = response.json()
            if isinstance(result, list):
                return result[0].get("translation_text", text)
            return result.get("translation_text", text)
            
        except requests.exceptions.RequestException as e:
            # Handle network errors and timeouts
            st.warning(f"Network error while connecting to Hugging Face API for translation: {str(e)}. Falling back to local model.")
            return self.translate_text_local(text)
    
    def translate_text_local(self, text, max_length=100):
        """Translate text using local model"""
        model, tokenizer = self.load_model()
        
        # Handle t5 models differently
        if "t5" in self.model_name.lower():
            # T5 requires a specific format for translation
            prefix = "translate English to "
            language = self.model_name.split("-")[-1].lower() if "-" in self.model_name else "German"
            input_text = f"{prefix}{language}: {text}"
        else:
            input_text = text
        
        # Tokenize and translate
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        
        # Decode and return translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def translate_text(self, text, use_api=True, max_length=100):
        """Translate text using specified method"""
        if use_api and self.api_key:
            try:
                return self.translate_text_api(text)
            except Exception as e:
                st.warning(f"Error using translation API: {str(e)}. Falling back to local model.")
                return self.translate_text_local(text, max_length)
        else:
            return self.translate_text_local(text, max_length)
            
    def translate_captions(self, captions, use_api=True):
        """Translate a list of captions"""
        translations = []
        for caption in captions:
            translation = self.translate_text(caption, use_api)
            translations.append(translation)
        return translations
    
    def translate_timestamped_captions(self, timestamped_captions, use_api=True):
        """Translate timestamped captions while preserving timestamps"""
        translated_timestamped_captions = []
        for timestamp, caption in timestamped_captions:
            translated_caption = self.translate_text(caption, use_api)
            translated_timestamped_captions.append((timestamp, translated_caption))
        return translated_timestamped_captions 