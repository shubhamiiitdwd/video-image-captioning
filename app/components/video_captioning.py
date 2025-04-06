import streamlit as st
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import io

from ..utils.video_captioning import VideoCaptioner
from ..utils.translation import CaptionTranslator
from ..utils.speech_recognition import SpeechRecognizer
from ..utils.config import (
    IMAGE_CAPTION_MODELS, TRANSLATION_MODELS, LANGUAGES,
    CAPTION_STYLES, DEFAULT_ADVANCED_SETTINGS, SPEECH_RECOGNITION_MODELS
)
from ..utils.exporter import (
    export_timestamped_captions_to_text,
    export_timestamped_captions_to_srt,
    get_download_link
)

def render_video_captioning(api_key, use_local_models):
    """Render video captioning component"""
    st.markdown("<h2 class='section-header'>🎬 Video Captioning</h2>", unsafe_allow_html=True)
    
    # Initialize state variables
    if "video_captions" not in st.session_state:
        st.session_state.video_captions = []
    if "translated_video_captions" not in st.session_state:
        st.session_state.translated_video_captions = []
    if "speech_transcripts" not in st.session_state:
        st.session_state.speech_transcripts = []
    if "video_captioner_settings" not in st.session_state:
        # Copy the default settings and adjust some video-specific values
        st.session_state.video_captioner_settings = DEFAULT_ADVANCED_SETTINGS.copy()
        st.session_state.video_captioner_settings["fps"] = 0.5
        st.session_state.video_captioner_settings["max_frames"] = 20
    if "speech_recognizer_settings" not in st.session_state:
        st.session_state.speech_recognizer_settings = {
            "beam_size": 5,
            "temperature": 0.0,
            "language": "en",
            "word_timestamps": True,
            "highlight_words": False,
            "initial_prompt": None
        }
    
    # Video uploader
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    # Display video if uploaded
    if uploaded_video is not None:
        st.video(uploaded_video)
    
    # Create two tabs for visual captioning and speech transcription
    tab1, tab2 = st.tabs(["🖼️ Visual Content", "🔊 Speech Transcription"])
    
    with tab1:
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select Captioning Model",
                options=list(IMAGE_CAPTION_MODELS.keys()),
                format_func=lambda x: IMAGE_CAPTION_MODELS[x],
                index=0,
                key="video_model"
            )
        
        with col2:
            caption_style = st.selectbox(
                "Caption Style",
                options=list(CAPTION_STYLES.keys()),
                format_func=lambda x: CAPTION_STYLES[x],
                index=0,
                key="video_caption_style"
            )
        
        # Frame extraction settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fps = st.slider(
                "Frames per second to process",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.video_captioner_settings.get("fps", 0.5),
                step=0.1,
                help="Lower values will process fewer frames but be faster"
            )
            st.session_state.video_captioner_settings["fps"] = fps
        
        with col2:
            max_frames = st.slider(
                "Maximum frames to process",
                min_value=5,
                max_value=60,
                value=st.session_state.video_captioner_settings.get("max_frames", 20),
                step=5,
                help="Limit the number of frames to process"
            )
            st.session_state.video_captioner_settings["max_frames"] = max_frames
            
        with col3:
            # Add option for keyframe extraction instead of fixed fps
            keyframe_mode = st.checkbox(
                "Smart Keyframe Extraction",
                value=st.session_state.video_captioner_settings.get("keyframe_mode", False),
                help="Extract only significant frames with scene changes"
            )
            st.session_state.video_captioner_settings["keyframe_mode"] = keyframe_mode
        
        # Translation options
        translation_model = st.selectbox(
            "Select Translation Model (optional)",
            options=["None"] + list(TRANSLATION_MODELS.keys()),
            format_func=lambda x: "No Translation" if x == "None" else TRANSLATION_MODELS[x],
            index=0,
            key="video_translation_model"
        )
        
        # Option to force local model
        force_local = st.checkbox("Force using local model (avoid API issues)", 
                                 value=use_local_models, 
                                 key="video_force_local")
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.markdown("### Caption Generation Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Update advanced settings
                st.session_state.video_captioner_settings["beam_size"] = st.slider(
                    "Beam Size",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.video_captioner_settings.get("beam_size", 5),
                    help="Higher values may improve quality but slower",
                    key="video_beam_size"
                )
                
                st.session_state.video_captioner_settings["temperature"] = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.video_captioner_settings.get("temperature", 1.0),
                    step=0.1,
                    help="Higher values make output more random",
                    key="video_temperature"
                )
                
            with col2:
                st.session_state.video_captioner_settings["max_length"] = st.slider(
                    "Max Caption Length",
                    min_value=10,
                    max_value=150,
                    value=st.session_state.video_captioner_settings.get("max_length", 50),
                    help="Maximum length of generated caption",
                    key="video_max_length"
                )
                
                st.session_state.video_captioner_settings["min_length"] = st.slider(
                    "Min Caption Length",
                    min_value=3,
                    max_value=50,
                    value=st.session_state.video_captioner_settings.get("min_length", 5),
                    help="Minimum length of generated caption",
                    key="video_min_length"
                )
            
            st.markdown("### Advanced Features")
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.video_captioner_settings["apply_post_processing"] = st.checkbox(
                    "Apply Post-Processing",
                    value=st.session_state.video_captioner_settings.get("apply_post_processing", True),
                    help="Clean and improve generated captions",
                    key="video_post_processing"
                )
                
                st.session_state.video_captioner_settings["enhanced_preprocessing"] = st.checkbox(
                    "Enhanced Frame Preprocessing",
                    value=st.session_state.video_captioner_settings.get("enhanced_preprocessing", False),
                    help="Apply image enhancements before captioning",
                    key="video_preprocessing"
                )
            
            with col2:
                st.session_state.video_captioner_settings["generate_summary"] = st.checkbox(
                    "Generate Video Summary",
                    value=st.session_state.video_captioner_settings.get("generate_summary", False),
                    help="Create a summary of the entire video content",
                    key="generate_summary"
                )
                
                st.session_state.video_captioner_settings["segment_scenes"] = st.checkbox(
                    "Scene Segmentation",
                    value=st.session_state.video_captioner_settings.get("segment_scenes", False),
                    help="Group captions by detected scenes",
                    key="segment_scenes"
                )
        
        # Generate captions button
        if uploaded_video is not None and st.button("Generate Visual Captions", key="gen_visual_captions"):
            try:
                # Create a temporary file to store the video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.getvalue())
                    video_path = tmp_file.name
                
                    # Initialize video captioner
                    video_captioner = VideoCaptioner(selected_model)
                    video_captioner.configure_advanced_settings(st.session_state.video_captioner_settings)
                    video_captioner.set_caption_style(caption_style)
                    
                    with st.spinner("Extracting frames and generating captions..."):
                        # Generate captions
                        captions = video_captioner.generate_video_caption(
                            video_path, fps=fps, max_frames=max_frames, use_api=not force_local
                        )
                        
                        # Store captions in session state
                        st.session_state.video_captions = captions
                        
                        # Display captions
                        st.markdown("<h3 class='sub-title'>Generated Visual Captions:</h3>", unsafe_allow_html=True)
                        
                        # Show timeline visualization
                        st.markdown("<h4>Timeline Visualization</h4>", unsafe_allow_html=True)
                        timeline_chart = create_timeline_visualization(captions)
                        st.pyplot(timeline_chart)
                        
                        # Display individual captions
                        for timestamp, caption in captions:
                            st.markdown(
                                f'<div class="video-caption-container">'
                                f'<span class="timestamp">[{timestamp}]</span> {caption}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Generate summary if enabled
                        if st.session_state.video_captioner_settings.get("generate_summary", False):
                            with st.spinner("Generating video summary..."):
                                summary = generate_video_summary(captions)
                                st.markdown("<h3 class='sub-title'>Video Summary:</h3>", unsafe_allow_html=True)
                                st.markdown(f'<div class="caption-text">{summary}</div>', unsafe_allow_html=True)
                        
                        # Translate if requested
                        if translation_model != "None":
                            with st.spinner("Translating captions..."):
                                translator = CaptionTranslator(translation_model)
                                translated_captions = translator.translate_timestamped_captions(
                                    captions, use_api=not force_local
                                )
                                
                                # Store translated captions
                                st.session_state.translated_video_captions = translated_captions
                                
                                # Display translated captions
                                st.markdown("<h3 class='sub-title'>Translated Captions:</h3>", unsafe_allow_html=True)
                                for timestamp, caption in translated_captions:
                                    st.markdown(
                                        f'<div class="video-caption-container translated-caption">'
                                        f'<span class="timestamp">[{timestamp}]</span> {caption}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                    
                    # Export options
                    if st.session_state.video_captions:
                        st.markdown("<h3 class='sub-title'>Export Options:</h3>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                    
                        with col1:
                            # Export as text
                            caption_text = export_timestamped_captions_to_text(st.session_state.video_captions)
                            st.download_button(
                                label="Download as Text",
                                data=caption_text,
                                file_name="video_captions.txt",
                                mime="text/plain",
                                key="download_txt"
                            )
                        
                        with col2:
                            # Export as SRT
                            srt_text = export_timestamped_captions_to_srt(st.session_state.video_captions)
                            st.download_button(
                                label="Download as SRT",
                                data=srt_text,
                                file_name="video_captions.srt",
                                mime="text/plain",
                                key="download_srt"
                            )
                        
                        # Export translated captions if available
                        if st.session_state.translated_video_captions:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export translated as text
                                translated_text = export_timestamped_captions_to_text(
                                    st.session_state.translated_video_captions
                                )
                                st.download_button(
                                    label="Download Translated Text",
                                    data=translated_text,
                                    file_name="translated_captions.txt",
                                    mime="text/plain",
                                    key="download_trans_txt"
                                )
                            
                            with col2:
                                # Export translated as SRT
                                translated_srt = export_timestamped_captions_to_srt(
                                    st.session_state.translated_video_captions
                                )
                                st.download_button(
                                    label="Download Translated SRT",
                                    data=translated_srt,
                                    file_name="translated_captions.srt",
                                    mime="text/plain",
                                    key="download_trans_srt"
                                )
            except Exception as e:
                st.error(f"Error generating captions: {str(e)}")
                
                # Try to clean up the temporary file
                try:
                    if 'video_path' in locals():
                        if os.path.exists(video_path):
                            os.remove(video_path)
                except Exception:
                    pass
    
    # Speech transcription tab
    with tab2:
        # Model selection for speech recognition
        speech_model = st.selectbox(
            "Select Speech Recognition Model",
            options=list(SPEECH_RECOGNITION_MODELS.keys()),
            format_func=lambda x: SPEECH_RECOGNITION_MODELS[x],
            index=1,  # Default to base model
            key="speech_model"
        )
        
        # Language selection
        language = st.selectbox(
            "Select Language (optional)",
            options=["Auto-detect"] + list(LANGUAGES.keys()),
            format_func=lambda x: "Auto-detect" if x == "Auto-detect" else LANGUAGES[x],
            index=0,
            key="speech_language"
        )
        
        # Set language in settings
        if language != "Auto-detect":
            st.session_state.speech_recognizer_settings["language"] = language
        else:
            st.session_state.speech_recognizer_settings["language"] = None
        
        # Advanced settings
        with st.expander("Advanced Speech Recognition Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.speech_recognizer_settings["beam_size"] = st.slider(
                    "Beam Size",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.speech_recognizer_settings.get("beam_size", 5),
                    help="Higher values may improve accuracy but slower",
                    key="speech_beam_size"
                )
                
                st.session_state.speech_recognizer_settings["temperature"] = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.speech_recognizer_settings.get("temperature", 0.0),
                    step=0.1,
                    help="Lower values improve deterministic results",
                    key="speech_temperature"
                )
            
            with col2:
                st.session_state.speech_recognizer_settings["word_timestamps"] = st.checkbox(
                    "Use Word-level Timestamps",
                    value=st.session_state.speech_recognizer_settings.get("word_timestamps", True),
                    help="Generate timestamps at the word level (more precise)",
                    key="word_timestamps"
                )
                
                initial_prompt = st.text_area(
                    "Initial Prompt (optional)",
                    value=st.session_state.speech_recognizer_settings.get("initial_prompt", ""),
                    height=100,
                    help="Optional text to guide the transcription",
                    key="initial_prompt"
                )
                st.session_state.speech_recognizer_settings["initial_prompt"] = initial_prompt if initial_prompt else None
        
        # Generate transcription button
        if uploaded_video is not None and st.button("Generate Speech Transcription", key="gen_speech_transcription"):
            try:
                # Create a temporary file to store the video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.getvalue())
                    video_path = tmp_file.name
                
                # Initialize speech recognizer
                speech_recognizer = SpeechRecognizer(speech_model)
                speech_recognizer.configure_settings({
                    "language": st.session_state.speech_recognizer_settings.get("language"),
                    "chunk_size": 30
                })
                
                with st.spinner("Extracting audio and transcribing speech..."):
                    # Generate transcription with live updates
                    st.info("Transcription in progress - you'll see updates as processing continues")
                    transcripts = speech_recognizer.transcribe_video(video_path)
                    
                    # Store transcripts in session state
                    st.session_state.speech_transcripts = transcripts
                    
                    # Display transcripts
                    st.markdown("<h3 class='sub-title'>Speech Transcription:</h3>", unsafe_allow_html=True)
                    
                    # Display individual segments
                    for start_time, end_time, text in transcripts:
                        st.markdown(
                            f'<div class="video-caption-container speech-transcript">'
                            f'<span class="timestamp">[{start_time} → {end_time}]</span> {text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                # Export options
                if st.session_state.speech_transcripts:
                    st.markdown("<h3 class='sub-title'>Export Options:</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export as text
                        transcript_text = export_speech_transcript_to_text(st.session_state.speech_transcripts)
                        st.download_button(
                            label="Download as Text",
                            data=transcript_text,
                            file_name="speech_transcript.txt",
                            mime="text/plain",
                            key="download_transcript_txt"
                        )
                    
                    with col2:
                        # Export as SRT
                        srt_text = export_speech_transcript_to_srt(st.session_state.speech_transcripts)
                        st.download_button(
                            label="Download as SRT",
                            data=srt_text,
                            file_name="speech_transcript.srt",
                            mime="text/plain",
                            key="download_transcript_srt"
                        )
                    
                    # Option to combine visual captions and speech transcripts
                    if st.session_state.video_captions and st.button("Combine Visual and Speech Captions"):
                        combined_captions = combine_captions_and_transcripts(
                            st.session_state.video_captions, 
                            st.session_state.speech_transcripts
                        )
                        
                        st.markdown("<h3 class='sub-title'>Combined Visual and Speech Captions:</h3>", unsafe_allow_html=True)
                        
                        for timestamp, caption in combined_captions:
                            st.markdown(
                                f'<div class="video-caption-container combined-caption">'
                                f'<span class="timestamp">[{timestamp}]</span> {caption}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Export combined captions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export as text
                            combined_text = export_timestamped_captions_to_text(combined_captions)
                            st.download_button(
                                label="Download Combined as Text",
                                data=combined_text,
                                file_name="combined_captions.txt",
                                mime="text/plain",
                                key="download_combined_txt"
                            )
                        
                        with col2:
                            # Export as SRT
                            combined_srt = export_timestamped_captions_to_srt(combined_captions)
                            st.download_button(
                                label="Download Combined as SRT",
                                data=combined_srt,
                                file_name="combined_captions.srt",
                                mime="text/plain",
                                key="download_combined_srt"
                            )
            except Exception as e:
                st.error(f"Error generating speech transcription: {str(e)}")
                
                # Try to clean up the temporary file
                try:
                    if 'video_path' in locals():
                        if os.path.exists(video_path):
                            os.remove(video_path)
                except Exception:
                    pass

def create_timeline_visualization(captions):
    """Create a visual timeline of captions"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Convert timestamps to minutes as floats
    times = []
    for timestamp, _ in captions:
        minutes, seconds = map(int, timestamp.split(':'))
        times.append(minutes + seconds/60)
    
    # Create a timeline
    ax.scatter(times, [1] * len(times), marker='o', s=100, color='#3498db')
    
    # Add caption text
    for i, (timestamp, caption) in enumerate(captions):
        truncated_caption = caption[:20] + '...' if len(caption) > 20 else caption
        ax.annotate(truncated_caption, (times[i], 1), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Set labels and title
    ax.set_yticks([])
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Caption Timeline')
    
    # Customize grid and styling
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Adjust spacing
    fig.tight_layout()
    
    # Convert plot to a displayable format
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    
    return fig

def generate_video_summary(captions):
    """Generate a summary of the video based on the captions"""
    if not captions:
        return "No captions available to generate summary."
    
    # Extract just the captions
    captions_text = [caption for _, caption in captions]
    
    if len(captions_text) >= 3:
        # Get beginning, middle, and end captions
        beginning = captions_text[0]
        middle = captions_text[len(captions_text)//2]
        end = captions_text[-1]
        
        summary = f"The video begins with {beginning.lower()} "
        summary += f"In the middle, {middle.lower()} "
        summary += f"Finally, {end.lower()}"
        
        return summary
    else:
        # If very few captions, just combine them
        return " ".join(captions_text)

def export_speech_transcript_to_text(transcript):
    """Export speech transcript to a text file format"""
    output = []
    for start_time, end_time, text in transcript:
        output.append(f"[{start_time} → {end_time}] {text}")
    return "\n".join(output)

def export_speech_transcript_to_srt(transcript):
    """Export speech transcript to SRT subtitle format"""
    srt_output = []
    
    for i, (start_time, end_time, text) in enumerate(transcript, 1):
        # Convert from MM:SS.ms format to SRT format (00:MM:SS,ms)
        start_mins, start_secs = start_time.split(":")
        end_mins, end_secs = end_time.split(":")
        
        start_srt = f"00:{start_mins}:{start_secs.replace('.', ',')}"
        end_srt = f"00:{end_mins}:{end_secs.replace('.', ',')}"
        
        srt_output.append(f"{i}")
        srt_output.append(f"{start_srt} --> {end_srt}")
        srt_output.append(f"{text}")
        srt_output.append("")
    
    return "\n".join(srt_output)

def combine_captions_and_transcripts(visual_captions, speech_transcripts):
    """Combine visual captions and speech transcripts into a unified timeline"""
    combined = []
    
    # Process visual captions
    for timestamp, caption in visual_captions:
        combined.append((timestamp, f"[Visual] {caption}"))
    
    # Process speech transcripts
    for start_time, _, text in speech_transcripts:
        combined.append((start_time, f"[Speech] {text}"))
    
    # Sort by timestamp
    combined.sort(key=lambda x: x[0])
    
    return combined 