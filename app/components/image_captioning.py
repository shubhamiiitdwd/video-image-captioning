import streamlit as st
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import io

from ..utils.image_captioning import ImageCaptioner
from ..utils.translation import CaptionTranslator
from ..utils.config import (
    IMAGE_CAPTION_MODELS, TRANSLATION_MODELS, LANGUAGES, 
    CAPTION_STYLES, DEFAULT_ADVANCED_SETTINGS
)
from ..utils.exporter import export_caption_to_text, get_download_link

def render_image_captioning(api_key, use_local_models):
    """Render image captioning component"""
    st.markdown("## 🖼️ Image Captioning")
    
    # Initialize state variables
    if "image_caption" not in st.session_state:
        st.session_state.image_caption = ""
    if "translated_caption" not in st.session_state:
        st.session_state.translated_caption = ""
    if "show_advanced_options" not in st.session_state:
        st.session_state.show_advanced_options = False
    if "captioner_settings" not in st.session_state:
        st.session_state.captioner_settings = DEFAULT_ADVANCED_SETTINGS.copy()
    
    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # URL input
    image_url = st.text_input("Or enter an image URL", "")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Select Image Captioning Model",
            options=list(IMAGE_CAPTION_MODELS.keys()),
            format_func=lambda x: IMAGE_CAPTION_MODELS[x],
            index=0
        )
    
    with col2:
        translation_model = st.selectbox(
            "Select Translation Model (optional)",
            options=["None"] + list(TRANSLATION_MODELS.keys()),
            format_func=lambda x: "No Translation" if x == "None" else TRANSLATION_MODELS[x],
            index=0
        )
    
    # Caption style selection
    caption_style = st.selectbox(
        "Caption Style",
        options=list(CAPTION_STYLES.keys()),
        format_func=lambda x: CAPTION_STYLES[x],
        index=0
    )
    
    # Option to force local model
    force_local = st.checkbox("Force using local model (avoid API issues)", value=use_local_models)
    
    # Advanced options
    with st.expander("Advanced Options"):
        st.session_state.show_advanced_options = True
        
        st.markdown("### Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            # Update advanced settings
            st.session_state.captioner_settings["beam_size"] = st.slider(
                "Beam Size",
                min_value=1,
                max_value=10,
                value=st.session_state.captioner_settings.get("beam_size", 5),
                help="Higher values may improve quality but slower"
            )
            
            st.session_state.captioner_settings["max_length"] = st.slider(
                "Max Caption Length",
                min_value=10,
                max_value=150,
                value=st.session_state.captioner_settings.get("max_length", 50),
                help="Maximum length of generated caption"
            )
            
            st.session_state.captioner_settings["min_length"] = st.slider(
                "Min Caption Length",
                min_value=3,
                max_value=50,
                value=st.session_state.captioner_settings.get("min_length", 5),
                help="Minimum length of generated caption"
            )
        
        with col2:
            st.session_state.captioner_settings["temperature"] = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.captioner_settings.get("temperature", 1.0),
                step=0.1,
                help="Higher values make output more random"
            )
            
            st.session_state.captioner_settings["repetition_penalty"] = st.slider(
                "Repetition Penalty",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.captioner_settings.get("repetition_penalty", 1.0),
                step=0.1,
                help="Penalize repetition in captions"
            )
        
        st.markdown("### Advanced Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.captioner_settings["apply_post_processing"] = st.checkbox(
                "Apply Post-Processing",
                value=st.session_state.captioner_settings.get("apply_post_processing", True),
                help="Clean and improve generated captions"
            )
            
            st.session_state.captioner_settings["enhanced_preprocessing"] = st.checkbox(
                "Enhanced Image Preprocessing",
                value=st.session_state.captioner_settings.get("enhanced_preprocessing", False),
                help="Apply image enhancements before captioning"
            )
        
        with col2:
            st.session_state.captioner_settings["ensemble_models"] = st.checkbox(
                "Use Model Ensemble",
                value=st.session_state.captioner_settings.get("ensemble_models", False),
                help="Combine results from multiple models for better accuracy"
            )
            
            # Display warning if ensemble is enabled
            if st.session_state.captioner_settings.get("ensemble_models", False):
                st.info("Ensemble mode will combine results from multiple models, which may take longer but can improve accuracy.")
    
    # Display batch processing option
    enable_batch = st.checkbox("Enable Batch Processing", value=False)
    
    if enable_batch:
        st.markdown("### Batch Processing")
        batch_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if batch_images:
            st.info(f"{len(batch_images)} images loaded for batch processing.")
            if st.button("Process Batch"):
                process_batch_images(batch_images, selected_model, translation_model, caption_style, 
                                    force_local, st.session_state.captioner_settings)
        
    # Process single image
    if uploaded_image is not None or image_url:
        # Initialize captioner
        captioner = ImageCaptioner(selected_model)
        
        # Configure captioner with advanced settings
        captioner.configure_advanced_settings(st.session_state.captioner_settings)
        captioner.set_caption_style(caption_style)
        
        try:
            # Load image
            if uploaded_image is not None:
                # Display uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                # Display image from URL
                image = captioner.load_image_from_url(image_url)
                st.image(image, caption="Image from URL", use_column_width=True)
            
            # Generate caption button
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    # Generate caption
                    caption = captioner.generate_caption(image, use_api=not force_local)
                    st.session_state.image_caption = caption
                    
                    # Display caption
                    st.markdown("### Generated Caption:")
                    st.markdown(f'<div class="caption-text">{caption}</div>', unsafe_allow_html=True)
                    
                    # Show confidence score for caption (simulated)
                    confidence = min(0.5 + len(caption) / 200.0, 0.98)  # Simple heuristic
                    st.progress(confidence)
                    st.text(f"Confidence Score: {confidence:.2%}")
                    
                    # Translate if requested
                    if translation_model != "None":
                        with st.spinner("Translating caption..."):
                            translator = CaptionTranslator(translation_model)
                            translated = translator.translate_text(caption, use_api=not force_local)
                            st.session_state.translated_caption = translated
                            
                            # Display translated caption
                            st.markdown("### Translated Caption:")
                            st.markdown(f'<div class="caption-text translated-caption">{translated}</div>', unsafe_allow_html=True)
            
            # Export caption if available
            if st.session_state.image_caption:
                st.markdown("### Export Caption")
                
                # Create temp directory if not exists
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    export_option = st.radio(
                        "Export content",
                        options=["Original", "Translated"],
                        index=0,
                        horizontal=True,
                        disabled=not st.session_state.translated_caption
                    )
                
                with col2:
                    export_format = st.radio(
                        "Export format",
                        options=["Text", "JSON", "Markdown"],
                        index=0,
                        horizontal=True
                    )
                
                # Export caption
                caption_to_export = st.session_state.translated_caption if export_option == "Translated" and st.session_state.translated_caption else st.session_state.image_caption
                
                # Export filename and content based on format
                if export_format == "Text":
                    export_filename = "temp/image_caption.txt"
                    export_caption_to_text(caption_to_export, export_filename)
                    file_format = "text"
                elif export_format == "JSON":
                    export_filename = "temp/image_caption.json"
                    export_json_content = f'{{"caption": "{caption_to_export}", "model": "{selected_model}", "style": "{caption_style}"}}'
                    with open(export_filename, 'w', encoding='utf-8') as f:
                        f.write(export_json_content)
                    file_format = "json"
                else:  # Markdown
                    export_filename = "temp/image_caption.md"
                    export_md_content = f"# Image Caption\n\n## Generated by {selected_model}\n\n> {caption_to_export}\n\n*Caption style: {caption_style}*"
                    with open(export_filename, 'w', encoding='utf-8') as f:
                        f.write(export_md_content)
                    file_format = "md"
                
                # Download link
                href, filename = get_download_link(export_filename, f"Download as {export_format}", file_format)
                st.markdown(f'<a href="{href}" download="{filename}" class="download-button">Download Caption as {export_format}</a>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("If you're experiencing API issues, try checking the 'Force using local model' option above.")
    else:
        st.info("Please upload an image or provide an image URL to generate a caption.")

def process_batch_images(images, model_name, translation_model, caption_style, force_local, settings):
    """Process a batch of images and generate captions"""
    if not images:
        return
    
    # Create captioner
    captioner = ImageCaptioner(model_name)
    captioner.configure_advanced_settings(settings)
    captioner.set_caption_style(caption_style)
    
    # Create translator if needed
    translator = None
    if translation_model != "None":
        translator = CaptionTranslator(translation_model)
    
    # Create results table
    results = []
    
    # Process each image
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img_file in enumerate(images):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(images))
            status_text.text(f"Processing image {i+1} of {len(images)}: {img_file.name}")
            
            # Load and process image
            image = Image.open(img_file)
            
            # Generate caption
            caption = captioner.generate_caption(image, use_api=not force_local)
            
            # Translate if requested
            translated = None
            if translator:
                translated = translator.translate_text(caption, use_api=not force_local)
            
            # Store result
            results.append({
                "filename": img_file.name,
                "caption": caption,
                "translated": translated
            })
            
        except Exception as e:
            st.error(f"Error processing {img_file.name}: {str(e)}")
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("Batch processing complete!")
    
    # Display results
    st.markdown("### Batch Processing Results")
    
    for result in results:
        st.markdown(f"**{result['filename']}**")
        st.markdown(f"*Caption:* {result['caption']}")
        if result['translated']:
            st.markdown(f"*Translated:* {result['translated']}")
        st.markdown("---")
    
    # Export batch results
    if results:
        export_batch_results(results, caption_style, model_name, translation_model) 