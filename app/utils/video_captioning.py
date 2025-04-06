import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from .image_captioning import ImageCaptioner
from .config import CAPTION_PROMPTS
import os

class VideoCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.image_captioner = ImageCaptioner(model_name)
        self.caption_style = "default"
        self.advanced_settings = {
            "beam_size": 5,
            "max_length": 50,
            "min_length": 5,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "fps": 0.5,
            "max_frames": 20,
            "apply_post_processing": True,
            "enhanced_preprocessing": False,
            "keyframe_mode": False,
            "segment_scenes": False,
            "generate_summary": False
        }
    
    def configure_advanced_settings(self, settings):
        """Apply advanced settings"""
        # Update local settings
        for key, value in settings.items():
            if key in self.advanced_settings:
                self.advanced_settings[key] = value
        
        # Also update the image captioner settings
        self.image_captioner.configure_advanced_settings(settings)
    
    def set_caption_style(self, style):
        """Set the caption style"""
        self.caption_style = style
        self.image_captioner.set_caption_style(style)
        
    def extract_frames(self, video_path, fps=None, max_frames=None):
        """Extract frames from video at specified FPS or using keyframe detection"""
        if fps is None:
            fps = self.advanced_settings.get("fps", 0.5)
            
        if max_frames is None:
            max_frames = self.advanced_settings.get("max_frames", 20)
            
        keyframe_mode = self.advanced_settings.get("keyframe_mode", False)
        
        frames = []
        temp_frames = []
        
        # Verify that the video file exists and get absolute path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        video_path = os.path.abspath(video_path)
        
        try:
            # Load video using moviepy
            clip = VideoFileClip(video_path)
            
            if keyframe_mode:
                # Use OpenCV to detect keyframes
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video with OpenCV: {video_path}")
                    
                prev_frame = None
                frame_count = 0
                extracted_count = 0
                scene_threshold = 30.0  # Threshold for scene change detection
                
                while cap.isOpened() and extracted_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Only process every few frames for efficiency
                    if frame_count % int(clip.fps / 2) != 0:
                        continue
                    
                    # Convert to grayscale for comparison
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is None:
                        # Always include the first frame
                        prev_frame = gray
                        frame_time = frame_count / clip.fps
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        frames.append(pil_img)
                        temp_frames.append((frame_time, pil_img))
                        extracted_count += 1
                        continue
                    
                    # Calculate difference between frames
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    diff_score = np.mean(frame_diff)
                    
                    # If difference is significant, it's a keyframe
                    if diff_score > scene_threshold:
                        frame_time = frame_count / clip.fps
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # Apply preprocessing if enabled
                        if self.advanced_settings.get("enhanced_preprocessing", False):
                            pil_img = self.image_captioner.preprocess_image(pil_img)
                            
                        frames.append(pil_img)
                        temp_frames.append((frame_time, pil_img))
                        extracted_count += 1
                        prev_frame = gray
                
                cap.release()
                
            else:
                # Calculate frame skipping based on fps
                total_frames = int(clip.fps * clip.duration)
                skip_frames = int(clip.fps / fps)
                
                # Limit the number of frames
                num_frames = min(int(total_frames / skip_frames), max_frames)
                
                # Extract frames using tqdm for progress display
                for i in tqdm(range(0, num_frames)):
                    frame_idx = i * skip_frames
                    frame_time = frame_idx / clip.fps
                    
                    if frame_time < clip.duration:
                        frame = clip.get_frame(frame_time)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # Apply preprocessing if enabled
                        if self.advanced_settings.get("enhanced_preprocessing", False):
                            pil_img = self.image_captioner.preprocess_image(pil_img)
                            
                        frames.append(pil_img)
                        temp_frames.append((frame_time, pil_img))
            
            # Make sure to properly close the clip
            clip.close()
            
        except Exception as e:
            # Make sure the clip is closed if an error occurs
            try:
                if 'clip' in locals() and clip is not None:
                    clip.close()
            except:
                pass
                
            # Re-raise the exception with more context
            error_message = str(e)
            if "imageio_ffmpeg" in error_message or "ffmpeg" in error_message.lower():
                error_message = f"FFmpeg error in video processing: {error_message}"
            elif "[WinError 2]" in error_message or "system cannot find the file" in error_message.lower():
                error_message = f"File not found error: {error_message}"
                
            raise RuntimeError(f"Error extracting frames: {error_message}")
        
        # If frame segmentation is enabled, organize frames into scenes
        if self.advanced_settings.get("segment_scenes", False) and len(frames) > 3:
            # Simple scene segmentation - group frames into sets with significant time gaps
            segmented_frames = []
            segmented_temp_frames = []
            
            current_scene = [frames[0]]
            current_scene_temp = [temp_frames[0]]
            
            for i in range(1, len(temp_frames)):
                prev_time = temp_frames[i-1][0]
                curr_time = temp_frames[i][0]
                
                # If time gap is significant, start a new scene
                if curr_time - prev_time > 5.0:  # 5 seconds gap threshold
                    # Add representative frame from current scene
                    if current_scene:
                        segmented_frames.append(current_scene[len(current_scene)//2])
                        segmented_temp_frames.append(current_scene_temp[len(current_scene_temp)//2])
                        
                    # Start new scene
                    current_scene = [frames[i]]
                    current_scene_temp = [temp_frames[i]]
                else:
                    # Continue current scene
                    current_scene.append(frames[i])
                    current_scene_temp.append(temp_frames[i])
            
            # Add last scene
            if current_scene:
                segmented_frames.append(current_scene[len(current_scene)//2])
                segmented_temp_frames.append(current_scene_temp[len(current_scene_temp)//2])
                
            return segmented_frames, segmented_temp_frames
            
        return frames, temp_frames
    
    def generate_captions_for_frames(self, frames, use_api=True):
        """Generate captions for each extracted frame"""
        captions = []
        
        # Pass advanced settings to the captioner
        for setting, value in self.advanced_settings.items():
            if setting in self.image_captioner.advanced_settings:
                self.image_captioner.advanced_settings[setting] = value
        
        for frame in tqdm(frames):
            caption = self.image_captioner.generate_caption(frame, use_api=use_api)
            captions.append(caption)
        return captions
    
    def generate_video_caption(self, video_path, fps=None, max_frames=None, use_api=True):
        """Generate captions for a video by extracting frames and captioning them"""
        # Extract frames
        frames, timestamped_frames = self.extract_frames(video_path, fps, max_frames)
        
        # Generate captions for frames
        frame_captions = self.generate_captions_for_frames(frames, use_api)
        
        # Combine captions with timestamps
        timestamped_captions = []
        for i, (time, _) in enumerate(timestamped_frames):
            minutes = int(time / 60)
            seconds = int(time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            timestamped_captions.append((timestamp, frame_captions[i]))
        
        return timestamped_captions
    
    def generate_video_summary(self, timestamped_captions):
        """Generate a summary of the video based on the captions"""
        if not timestamped_captions:
            return "No captions available to generate summary."
            
        # Extract just the captions
        captions = [caption for _, caption in timestamped_captions]
        
        if len(captions) >= 3:
            # Get beginning, middle, and end captions
            beginning = captions[0]
            middle = captions[len(captions)//2]
            end = captions[-1]
            
            summary = f"The video begins with {beginning.lower()} "
            summary += f"In the middle, {middle.lower()} "
            summary += f"Finally, {end.lower()}"
            
            return summary
        else:
            # If very few captions, just combine them
            return " ".join(captions)
    
    def get_video_info(self, video_path):
        """Get basic information about the video"""
        clip = VideoFileClip(video_path)
        info = {
            "duration": clip.duration,
            "fps": clip.fps,
            "size": clip.size,
            "rotation": clip.rotation
        }
        clip.close()
        return info 