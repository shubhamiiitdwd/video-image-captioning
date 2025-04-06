import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import re
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, 
    AutoProcessor, AutoModelForCausalLM,
    Blip2Processor, Blip2ForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from .config import get_huggingface_api_key, CAPTION_PROMPTS
import streamlit as st

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.api_key = get_huggingface_api_key()
        self.caption_style = "default"
        self.advanced_settings = {
            "beam_size": 5,
            "max_length": 50,
            "min_length": 5,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "apply_post_processing": True,
            "enhanced_preprocessing": False,
            "ensemble_models": False
        }
        
    def configure_advanced_settings(self, settings):
        """Apply advanced settings"""
        for key, value in settings.items():
            if key in self.advanced_settings:
                self.advanced_settings[key] = value
                
    def set_caption_style(self, style):
        """Set the caption style"""
        self.caption_style = style
        
    def preprocess_image(self, image):
        """Apply enhanced preprocessing to image if enabled"""
        if not self.advanced_settings.get("enhanced_preprocessing", False):
            return image
            
        # Convert to numpy array if not already
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Apply basic image enhancements
        # 1. Normalize
        img_array = img_array.astype(np.float32) / 255.0
        
        # 2. Adjust contrast
        img_array = np.clip((img_array - 0.5) * 1.2 + 0.5, 0, 1)
        
        # 3. Convert back to uint8
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_array)
        
        return enhanced_image
        
    def load_model(self):
        """Load the selected model"""
        if self.model is None:
            try:
                if "blip2" in self.model_name.lower():
                    self.processor = Blip2Processor.from_pretrained(self.model_name)
                    self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name)
                elif "blip" in self.model_name.lower():
                    self.processor = BlipProcessor.from_pretrained(self.model_name)
                    self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                elif "git" in self.model_name.lower():
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                elif "vit-gpt2" in self.model_name.lower():
                    self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                    self.processor = {
                        "image_processor": ViTImageProcessor.from_pretrained(self.model_name),
                        "tokenizer": AutoTokenizer.from_pretrained(self.model_name)
                    }
                elif "ofa" in self.model_name.lower():
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                else:
                    # Default to BLIP if model type unknown
                    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                
                self.model = self.model.to(self.device)
                
            except Exception as e:
                st.error(f"Error loading model {self.model_name}: {str(e)}")
                # Fallback to BLIP base model
                st.warning(f"Falling back to BLIP base model")
                self.model_name = "Salesforce/blip-image-captioning-base"
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                
        return self.model, self.processor
    
    def generate_caption_api(self, image, max_new_tokens=50):
        """Generate caption using Hugging Face API"""
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
        
        # Apply preprocessing if enabled
        if self.advanced_settings.get("enhanced_preprocessing", False):
            image = self.preprocess_image(image)
        
        # Convert PIL image to bytes
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image  # Assume it's already bytes
        
        # API endpoint
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Get prompt based on caption style
        prompt = CAPTION_PROMPTS.get(self.caption_style, "")
        
        try:
            # API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Add parameters for the API request
            payload = {
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            # Add prompt if provided
            if prompt:
                payload["inputs"] = prompt
                
            # Make API call
            response = requests.post(
                API_URL, 
                headers=headers, 
                data=image_bytes,
                params=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                # If API is down (503), fall back to local model
                if response.status_code == 503:
                    st.warning("Hugging Face API is currently unavailable. Falling back to local model.")
                    return self.generate_caption_local(image, max_new_tokens)
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # Parse and return the result
            result = response.json()
            if isinstance(result, list):
                caption = result[0].get("generated_text", "Failed to generate caption")
            else:
                caption = result.get("generated_text", "Failed to generate caption")
                
            # Apply post-processing if enabled
            if self.advanced_settings.get("apply_post_processing", True):
                caption = self.post_process_caption(caption)
                
            return caption
            
        except requests.exceptions.RequestException as e:
            # Handle network errors and timeouts
            st.warning(f"Network error while connecting to Hugging Face API: {str(e)}. Falling back to local model.")
            return self.generate_caption_local(image, max_new_tokens)
    
    def post_process_caption(self, caption):
        """Post-process the generated caption to improve quality"""
        if not caption:
            return caption
            
        # Remove redundant whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        # Ensure first letter is capitalized
        if caption and len(caption) > 0:
            caption = caption[0].upper() + caption[1:]
        
        # Ensure proper ending punctuation
        if caption and not caption[-1] in ['.', '!', '?']:
            caption += '.'
            
        # Remove any residual special tokens
        caption = re.sub(r'<[^>]+>', '', caption)
        
        return caption
    
    def generate_caption_local(self, image, max_new_tokens=None):
        """Generate caption using local model"""
        # Use settings from advanced configuration
        if max_new_tokens is None:
            max_new_tokens = self.advanced_settings.get("max_length", 50)
            
        beam_size = self.advanced_settings.get("beam_size", 5)
        min_length = self.advanced_settings.get("min_length", 5)
        repetition_penalty = self.advanced_settings.get("repetition_penalty", 1.0)
        temperature = self.advanced_settings.get("temperature", 1.0)
        
        # Get prompt based on caption style
        prompt = CAPTION_PROMPTS.get(self.caption_style, "")
        
        # Apply preprocessing if enabled
        if self.advanced_settings.get("enhanced_preprocessing", False):
            image = self.preprocess_image(image)
        
        model, processor = self.load_model()
        
        try:
            if "blip2" in self.model_name.lower():
                # BLIP-2 model processing
                if prompt:
                    inputs = processor(image, prompt, return_tensors="pt").to(self.device)
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_length=min_length,
                        num_beams=beam_size,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature
                    )
                else:
                    inputs = processor(image, return_tensors="pt").to(self.device)
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_length=min_length,
                        num_beams=beam_size,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature
                    )
                caption = processor.decode(output[0], skip_special_tokens=True)
                
            elif "blip" in self.model_name.lower():
                # BLIP model processing
                if prompt:
                    inputs = processor(image, prompt, return_tensors="pt").to(self.device)
                else:
                    inputs = processor(image, return_tensors="pt").to(self.device)
                output = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature
                )
                caption = processor.decode(output[0], skip_special_tokens=True)
                
            elif "git" in self.model_name.lower():
                # GIT model processing
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values, 
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature
                )
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            elif "vit-gpt2" in self.model_name.lower():
                # ViT-GPT2 model processing
                img_processor = processor["image_processor"]
                tokenizer = processor["tokenizer"]
                
                pixel_values = img_processor(image, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = model.generate(
                    pixel_values, 
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature
                )
                caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
            else:
                # Default processing
                inputs = processor(image, return_tensors="pt").to(self.device)
                output = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    num_beams=beam_size,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature
                )
                caption = processor.decode(output[0], skip_special_tokens=True)
            
            # Apply post-processing if enabled
            if self.advanced_settings.get("apply_post_processing", True):
                caption = self.post_process_caption(caption)
                
            return caption
            
        except Exception as e:
            st.error(f"Error generating caption with {self.model_name}: {str(e)}")
            # Use a simpler model as fallback
            if self.model_name != "Salesforce/blip-image-captioning-base":
                st.warning("Falling back to BLIP base model")
                self.model_name = "Salesforce/blip-image-captioning-base"
                self.model = None
                self.processor = None
                return self.generate_caption_local(image, max_new_tokens)
            else:
                return "Failed to generate caption due to model error."
    
    def generate_ensemble_caption(self, image, use_api=True, models=None):
        """Generate captions using multiple models and combine them"""
        if models is None:
            # Default ensemble uses these three diverse models
            models = [
                "Salesforce/blip-image-captioning-base",
                "nlpconnect/vit-gpt2-image-captioning",
                "microsoft/git-base-coco"
            ]
        
        original_model = self.model_name
        captions = []
        
        for model_name in models:
            try:
                self.model_name = model_name
                self.model = None
                self.processor = None
                
                if use_api and self.api_key:
                    caption = self.generate_caption_api(image)
                else:
                    caption = self.generate_caption_local(image)
                    
                captions.append(caption)
            except Exception as e:
                st.warning(f"Model {model_name} failed: {str(e)}")
        
        # Restore original model
        self.model_name = original_model
        self.model = None
        self.processor = None
        
        # Return the longest caption as it likely has the most information
        if captions:
            # Different combination strategies can be implemented here
            # 1. Longest caption
            # return max(captions, key=len)
            
            # 2. Caption with median length (likely most balanced)
            return sorted(captions, key=len)[len(captions)//2]
        else:
            return "No models in the ensemble were able to generate a caption."
    
    def generate_caption(self, image, use_api=True, max_new_tokens=None):
        """Generate caption for an image"""
        # Use ensemble method if enabled
        if self.advanced_settings.get("ensemble_models", False):
            return self.generate_ensemble_caption(image, use_api)
            
        if use_api and self.api_key:
            try:
                return self.generate_caption_api(image, max_new_tokens)
            except Exception as e:
                st.warning(f"Error using API: {str(e)}. Falling back to local model.")
                return self.generate_caption_local(image, max_new_tokens)
        else:
            return self.generate_caption_local(image, max_new_tokens)
            
    def load_image_from_url(self, url):
        """Load image from URL"""
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    
    def load_image_from_file(self, file_bytes):
        """Load image from file bytes"""
        image = Image.open(BytesIO(file_bytes))
        return image 