import os
import base64
import json
import csv
from io import BytesIO, StringIO
import streamlit as st

def export_caption_to_text(caption, filename="caption.txt"):
    """Export a single caption to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(caption)
    return filename

def export_captions_to_text(captions, filename="captions.txt"):
    """Export a list of captions to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, caption in enumerate(captions, 1):
            f.write(f"Caption {i}: {caption}\n\n")
    return filename

def export_timestamped_captions_to_text(timestamped_captions, filename="video_captions.txt"):
    """Export timestamped captions to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for timestamp, caption in timestamped_captions:
            f.write(f"[{timestamp}] {caption}\n\n")
    return filename

def export_timestamped_captions_to_srt(timestamped_captions, filename="captions.srt"):
    """Export timestamped captions to an SRT subtitle file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (timestamp, caption) in enumerate(timestamped_captions, 1):
            minutes, seconds = timestamp.split(":")
            start_time = f"00:{minutes}:{seconds},000"
            
            # Calculate end time (add 5 seconds)
            min_val = int(minutes)
            sec_val = int(seconds) + 5
            if sec_val >= 60:
                min_val += 1
                sec_val -= 60
            end_time = f"00:{min_val:02d}:{sec_val:02d},000"
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{caption}\n\n")
    return filename

def export_batch_results(results, caption_style, model_name, translation_model):
    """Export batch processing results in various formats"""
    # Create temp directory if not exists
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Create a tab layout for different export formats
    tab1, tab2, tab3 = st.tabs(["CSV", "JSON", "Text"])
    
    # CSV export
    with tab1:
        st.markdown("### Export as CSV")
        
        # Create CSV content
        csv_file = StringIO()
        fieldnames = ["filename", "caption"]
        
        if any(result.get("translated") for result in results):
            fieldnames.append("translated")
            
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                "filename": result["filename"],
                "caption": result["caption"]
            }
            if "translated" in fieldnames and result.get("translated"):
                row["translated"] = result["translated"]
            writer.writerow(row)
        
        # Save to file
        csv_content = csv_file.getvalue()
        export_filename = "temp/batch_captions.csv"
        with open(export_filename, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        # Create download link
        href, filename = get_download_link(export_filename, "Download CSV", "csv")
        st.markdown(f'<a href="{href}" download="{filename}" class="download-button">Download as CSV</a>', unsafe_allow_html=True)
    
    # JSON export
    with tab2:
        st.markdown("### Export as JSON")
        
        # Create JSON with metadata
        json_data = {
            "metadata": {
                "model": model_name,
                "caption_style": caption_style,
                "translation_model": translation_model if translation_model != "None" else None,
                "count": len(results)
            },
            "captions": results
        }
        
        # Save to file
        export_filename = "temp/batch_captions.json"
        with open(export_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        
        # Create download link
        href, filename = get_download_link(export_filename, "Download JSON", "json")
        st.markdown(f'<a href="{href}" download="{filename}" class="download-button">Download as JSON</a>', unsafe_allow_html=True)
    
    # Text export
    with tab3:
        st.markdown("### Export as Text")
        
        # Create plain text content
        export_filename = "temp/batch_captions.txt"
        with open(export_filename, 'w', encoding='utf-8') as f:
            f.write(f"BATCH CAPTION RESULTS\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Style: {caption_style}\n")
            if translation_model != "None":
                f.write(f"Translation: {translation_model}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            # Write each result
            for result in results:
                f.write(f"File: {result['filename']}\n")
                f.write(f"Caption: {result['caption']}\n")
                if result.get("translated"):
                    f.write(f"Translated: {result['translated']}\n")
                f.write("\n" + "-"*30 + "\n\n")
        
        # Create download link
        href, filename = get_download_link(export_filename, "Download Text", "text")
        st.markdown(f'<a href="{href}" download="{filename}" class="download-button">Download as Text</a>', unsafe_allow_html=True)

def get_download_link(file_path, link_text="Download", file_format="text"):
    """Generate a download link for a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64_data = base64.b64encode(data).decode()
    
    # Determine MIME type
    mime_types = {
        "text": "text/plain",
        "srt": "application/x-subrip",
        "json": "application/json",
        "csv": "text/csv",
        "md": "text/markdown"
    }
    mime_type = mime_types.get(file_format, "text/plain")
    
    # Generate download link
    file_name = os.path.basename(file_path)
    href = f'data:{mime_type};base64,{b64_data}'
    
    return href, file_name 