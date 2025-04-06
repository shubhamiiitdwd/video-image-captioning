# Media Captioning Hub

An advanced application for automatically generating captions for images and videos, with speech transcription and translation capabilities.

## Features

- **Image Captioning**: Generate accurate descriptions for images using state-of-the-art AI models.
- **Video Captioning**: Extract frames from videos and generate captions with timestamps.
- **Speech Transcription**: Extract and transcribe speech from videos with precise timestamps.
- **Translation**: Translate captions to multiple languages.
- **Export Options**: Download captions as text files or SRT subtitles for videos.

## Requirements

- Python 3.8 or higher

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd media-captioning-hub
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Open your web browser and go to `http://localhost:8501`
2. Choose between image or video captioning tabs.
3. Upload an image or video file.
4. Configure the desired options (model, style, languages, etc.).
5. Generate captions and export them in your preferred format.

## Models

The application uses the following models:

### Image/Video Captioning
- BLIP (Base & Large)
- BLIP-2 (OPT-2.7B, OPT-6.7B, FLAN-T5-XL)
- GIT (Base & Large COCO)
- ViT-GPT2
- OFA Base
- CLIP+GPT

### Speech Recognition
- Whisper (Tiny, Base, Small, Medium, Large)

### Translation
- T5 Base
- M2M100 (418M & 1.2B)
- Various Helsinki-NLP models for language-specific translation

## License

[MIT License](LICENSE)

## Acknowledgements

- Hugging Face for providing the pre-trained models.
- Streamlit for the web app framework.
- OpenAI for the Whisper speech recognition model.

## Project Structure

```
media-captioning-hub/
├── app/
│   ├── components/         # UI components
│   ├── styles/             # CSS styling
│   └── utils/              # Helper utilities
├── .env                    # Environment variables 
├── app.py                  # Main application
├── requirements.txt        # Dependencies
└── README.md               # This documentation
```

## Advanced Configuration

- **Frame Extraction**: Adjust the frame rate and maximum frames to control video processing
- **Local Models**: Option to use models locally instead of API for offline usage
- **Custom Translations**: Select specific language pairs for caption translation

## Credits

This project utilizes models from Hugging Face's model hub, particularly:
- BLIP models from Salesforce
- GIT models from Microsoft
- Translation models from Helsinki-NLP and Google 