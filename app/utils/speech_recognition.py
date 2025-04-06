import os
import torch
import numpy as np
import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import shutil
import uuid
import moviepy.editor as mp
import soundfile as sf
import time

class SpeechRecognizer:
    def __init__(self, model_name="openai/whisper-base"):
        """
        Initialize speech recognizer using open source models from Hugging Face
        
        Parameters:
        -----------
        model_name: str
            Name of the Hugging Face model to use
            Default options: 
            - "openai/whisper-base" - Good balance of accuracy and speed
            - "openai/whisper-tiny" - Fastest option, less accurate
            - "facebook/wav2vec2-base-960h" - Alternative model for English only
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
        # Default settings
        self.settings = {
            "language": "en",         # Default language
            "chunk_size": 30,         # Process audio in chunks of N seconds
            "return_timestamps": True # Return word-level timestamps
        }
        
    def load_model(self):
        """Load the speech recognition model if not already loaded"""
        if self.model is None:
            try:
                st.info(f"Loading speech recognition model: {self.model_name}")
                
                if "whisper" in self.model_name:
                    # For Whisper models
                    self.processor = WhisperProcessor.from_pretrained(self.model_name)
                    self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                    
                    # Move model to appropriate device
                    self.model = self.model.to(self.device)
                else:
                    # For other models like Wav2Vec2, use the pipeline
                    self.asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_name,
                        device=self.device
                    )
                
                st.success(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Falling back to smaller Whisper model")
                
                # Fallback to tiny model which should work in most environments
                self.model_name = "openai/whisper-tiny"
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
    
    def configure_settings(self, settings):
        """Update settings for transcription"""
        for key, value in settings.items():
            if key in self.settings:
                self.settings[key] = value
    
    def extract_audio(self, video_path):
        """Extract audio from video file and return path to the audio file"""
        # Create a unique temp directory
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join(tempfile.gettempdir(), f"speech_rec_{unique_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Path for extracted audio
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        st.info(f"Extracting audio from video...")
        
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Check if video has audio
            if video.audio is None:
                st.warning("Video doesn't contain an audio track")
                # Create silent audio of the same duration
                sr = 16000  # Sample rate
                silence = np.zeros(int(video.duration * sr))
                sf.write(audio_path, silence, sr)
            else:
                # Extract audio
                video.audio.write_audiofile(
                    audio_path,
                    fps=16000,  # Sample rate for speech recognition
                    nbytes=2,   # 16-bit audio
                    codec='pcm_s16le',
                    ffmpeg_params=["-ac", "1"],  # Mono audio
                    logger=None  # Disable logging
                )
            
            # Close video to release resources
            video.close()
            
            return audio_path, temp_dir
            
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract audio from video: {str(e)}")
    
    def transcribe_with_timestamps(self, audio_path):
        """
        Transcribe audio with timestamps using Whisper model
        
        Parameters:
        -----------
        audio_path: str
            Path to the audio file
            
        Returns:
        --------
        List of tuples (start_time, end_time, text)
        """
        # Load model if not already loaded
        self.load_model()
        
        try:
            # Live progress display
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load audio file
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # For non-Whisper models that don't support timestamps well
            if "whisper" not in self.model_name:
                status_text.text("Transcribing with base model (no timestamps)")
                result = self.asr_pipeline(audio_array)
                
                # Create a simple segment without detailed timestamps
                text = result.get("text", "")
                if text:
                    return [(self.format_timestamp(0), self.format_timestamp(len(audio_array)/sr), text)]
                return []
            
            # Process with Whisper
            status_text.text("Processing audio with Whisper...")
            
            # Get language code if specified
            language = self.settings.get("language")
            forced_decoder_ids = None
            
            if language and language != "auto":
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
            
            # Process in smaller chunks for live updates
            chunk_size = self.settings.get("chunk_size", 30)  # in seconds
            sample_rate = 16000
            chunk_length = int(chunk_size * sample_rate)
            
            all_segments = []
            
            # Calculate total number of chunks
            total_chunks = int(np.ceil(len(audio_array) / chunk_length))
            
            # Process each chunk
            for i in range(total_chunks):
                # Update progress
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Transcribing chunk {i+1}/{total_chunks}...")
                
                # Extract chunk
                start_idx = i * chunk_length
                end_idx = min(start_idx + chunk_length, len(audio_array))
                chunk = audio_array[start_idx:end_idx]
                
                # Calculate time offset for this chunk
                time_offset = i * chunk_size
                
                # Process with Whisper
                input_features = self.processor(
                    chunk, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate tokens
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        return_timestamps=True
                    )
                
                # Decode the tokens with timestamps
                transcription = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=False
                )
                
                # Extract timestamp tokens and text
                segments = self.parse_whisper_timestamps(transcription[0], time_offset)
                all_segments.extend(segments)
                
                # Short pause to allow UI to update
                time.sleep(0.1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return all_segments
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            # Return error as transcript for visibility
            return [(self.format_timestamp(0), self.format_timestamp(10), 
                    f"Transcription error: {str(e)}")]
    
    def parse_whisper_timestamps(self, transcription, time_offset=0):
        """Parse Whisper special tokens to extract timestamps and text"""
        # Simple parsing for demonstration
        # In a full implementation, you would parse the <|starttime|> and <|endtime|> tokens
        # For now, we'll create segments based on words and punctuation
        
        # Remove special tokens and get plain text
        text = self.processor.tokenizer.decode(
            self.processor.tokenizer.encode(transcription), 
            skip_special_tokens=True
        )
        
        # Simple segment creation based on punctuation
        segments = []
        current_segment = []
        segment_start = time_offset
        
        # Split by punctuation that likely indicates sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Estimate duration based on characters (rough approximation)
        total_chars = len(text)
        char_duration = 0.1  # seconds per character (approx)
        
        char_count = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Approximate timestamps
            start_time = segment_start + (char_count / total_chars) * (len(text) * char_duration)
            char_count += len(sentence)
            end_time = segment_start + (char_count / total_chars) * (len(text) * char_duration)
            
            segments.append((
                self.format_timestamp(start_time),
                self.format_timestamp(end_time),
                sentence.strip()
            ))
        
        # If no segments were created, create one with the entire text
        if not segments and text:
            segments.append((
                self.format_timestamp(time_offset),
                self.format_timestamp(time_offset + len(text) * char_duration),
                text
            ))
            
        return segments
    
    def format_timestamp(self, seconds):
        """Format seconds to MM:SS.ms format"""
        seconds = float(seconds) if seconds is not None else 0
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:05.2f}"
    
    def transcribe_video(self, video_path):
        """
        Main method to transcribe speech in a video with timestamps
        
        Parameters:
        -----------
        video_path: str
            Path to the video file
            
        Returns:
        --------
        List of tuples (start_time, end_time, text)
        """
        # Check if file exists
        if not os.path.exists(video_path):
            st.error(f"Video file not found: {video_path}")
            return [(self.format_timestamp(0), self.format_timestamp(0), "Video file not found")]
        
        temp_dir = None
        try:
            # Extract audio first
            audio_path, temp_dir = self.extract_audio(video_path)
            
            # Transcribe with timestamps
            return self.transcribe_with_timestamps(audio_path)
            
        except Exception as e:
            st.error(f"Error in transcription process: {str(e)}")
            return [(self.format_timestamp(0), self.format_timestamp(0), f"Error: {str(e)}")]
            
        finally:
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    st.warning(f"Could not clean up temporary files: {str(e)}") 