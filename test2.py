from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import os
from pathlib import Path
import piexif
import exif
import tempfile
import pandas as pd
import google.generativeai as genai
from PIL import ImageOps, ExifTags, Image
import PIL


# Try to add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import pillow_heif

# ===============================
# API Configuration
# ===============================
def init_gemini():
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            key = f.read().strip()
            if key:
                genai.configure(api_key=key)
                return genai.GenerativeModel('gemini-2.5-flash-lite')#gemini-3-flash-preview gemini-2.5-flash gemini-2.5-flash-lite
    return None

model = init_gemini()


def extract_glucose_from_single_image(image_path):
    """
    Extract glucose reading from a single glucometer image.
    
    Args:
        image_path: Path to the image file (str or file-like object)
        
    Returns:
        str: Glucose reading value
    """
    try:
        # Load image
        if isinstance(image_path, str):
            img = PIL.Image.open(image_path)
        else:
            img = PIL.Image.open(image_path)
        
        # Correct orientation
        img = ImageOps.exif_transpose(img)
        
        # Extract glucose reading using Gemini
        prompt = """
        Analyze this glucometer image. Identify ONLY the primary glucose reading (the big number).
        Respond with just the number (e.g., '111'), nothing else.
        """
        response = model.generate_content([prompt, img])
        glucose_reading = response.text.strip()
        
        return glucose_reading
        
    except Exception as e:
        return f'Error: {str(e)}'

def extract_image_metadata_with_glucose(path, skip_existing_csv=None):
    """
    Extracts image metadata (date, time) and glucose reading for all images in a directory or a single image.
    For HEIC images, converts them to JPG first to extract metadata reliably.
    
    Args:
        path: Either a directory path or a single image file path
        skip_existing_csv: Optional path to CSV file with already processed images to skip
        
    Returns:
        If path is a file: pandas DataFrame with single row containing 'filename', 'date', 'time', 'glucose_reading'
        If path is a directory: pandas DataFrame with all images and their metadata
    """
    
    # Load existing filenames from CSV if provided
    existing_filenames = set()
    if skip_existing_csv and os.path.exists(skip_existing_csv):
        try:
            existing_df = pd.read_csv(skip_existing_csv)
            existing_filenames = set(existing_df['filename'].values)
            print(f"Loaded {len(existing_filenames)} existing filenames from {skip_existing_csv}")
        except Exception as e:
            print(f"Warning: Could not load existing CSV: {e}")
    
    def process_single_image_with_glucose(image_path):
        """Helper function to extract metadata and glucose reading from a single image"""
        temp_jpg_path = None
        filename = os.path.basename(image_path)
        
        try:
            # Check if it's a HEIC file and convert it
            if image_path.lower().endswith(('.heic', '.heif')):
                temp_jpg_path = convert_heic_to_jpg_with_metadata(image_path)
                if not temp_jpg_path:
                    return {
                        'filename': filename,
                        'date': None,
                        'time': None,
                        'glucose_reading': 'Error: Could not convert HEIC'
                    }
                image_to_process = temp_jpg_path
            else:
                image_to_process = image_path
            
            # Extract date and time
            datetime_str = None
            
            # Try exif library first
            try:
                with open(image_to_process, 'rb') as f:
                    image_exif = exif.Image(f)
                    if image_exif.has_exif:
                        datetime_str = image_exif.get("datetime_original") or image_exif.get("datetime")
            except:
                pass
            
            # Fallback to piexif
            if not datetime_str:
                try:
                    exif_dict = piexif.load(image_to_process)
                    exif_data = exif_dict.get("0th", {})
                    
                    # Look for DateTime tags
                    for tag in [306, 36867, 36868]:  # DateTime, DateTimeOriginal, DateTimeDigitized
                        if tag in exif_data:
                            value = exif_data[tag]
                            if isinstance(value, bytes):
                                datetime_str = value.decode('utf-8')
                            else:
                                datetime_str = str(value)
                            if datetime_str:
                                break
                except:
                    pass
            
            # Fallback to PIL
            if not datetime_str:
                try:
                    img = Image.open(image_to_process)
                    exif_data = img._getexif()
                    
                    if exif_data:
                        exif_dict = {
                            TAGS.get(tag, tag): value
                            for tag, value in exif_data.items()
                        }
                        datetime_str = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
                except:
                    pass
            
            date_str, time_str = None, None
            if datetime_str:
                try:
                    if isinstance(datetime_str, bytes):
                        datetime_str = datetime_str.decode('utf-8')
                    dt = datetime.strptime(str(datetime_str).strip(), "%Y:%m:%d %H:%M:%S")
                    date_str = dt.date().isoformat()
                    time_str = dt.time().isoformat()
                except:
                    pass
            
            # Extract glucose reading
            glucose_reading = extract_glucose_from_single_image(image_path)
            
            return {
                'filename': filename,
                'date': date_str,
                'time': time_str,
                'glucose_reading': glucose_reading
            }
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                'filename': filename,
                'date': None,
                'time': None,
                'glucose_reading': f'Error: {str(e)}'
            }
        
        finally:
            # Clean up temporary JPG file
            if temp_jpg_path and os.path.exists(temp_jpg_path):
                try:
                    os.remove(temp_jpg_path)
                except:
                    pass
    
    # Check if path is a directory
    if os.path.isdir(path):
        results = []
        
        # Process all files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            
            # Skip if already in existing CSV
            if filename in existing_filenames:
                print(f"Skipping {filename} (already processed)")
                continue
            
            # Check if it's a file
            if os.path.isfile(file_path):
                try:
                    # Try to open as an image
                    Image.open(file_path)
                    print(f"Processing {filename}...")
                    result = process_single_image_with_glucose(file_path)
                    results.append(result)
                except (IOError, OSError):
                    # Not an image file, skip it
                    pass
        
        # Return as DataFrame
        df = pd.DataFrame(results)
        return df
    
    # If it's a single file, return DataFrame with single row
    elif os.path.isfile(path):
        filename = os.path.basename(path)
        if filename in existing_filenames:
            print(f"Skipping {filename} (already processed)")
            return pd.DataFrame()
        result = process_single_image_with_glucose(path)
        df = pd.DataFrame([result])
        return df
    
    else:
        raise ValueError(f"Path does not exist: {path}")


def convert_heic_to_jpg_with_metadata(heic_path):
    """
    Converts a HEIC file to JPG format while preserving metadata.
    Returns the path to the temporary JPG file.
    """
    try:
        image = Image.open(heic_path)
        exif_bytes = image.info.get("exif")
        
        # Create a temporary JPG file
        temp_fd, temp_jpg_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        
        if exif_bytes:
            image.save(temp_jpg_path, "JPEG", exif=exif_bytes, quality=95)
        else:
            image.save(temp_jpg_path, "JPEG", quality=95)
        
        return temp_jpg_path
    except Exception as e:
        print(f"Error converting HEIC {heic_path}: {e}")
        return None


# Example usage with directory - skip already processed images
print("\nProcessing only new images (not in output_with_glucose.csv):")
df_new = extract_image_metadata_with_glucose(
    '/Users/richaanuragini/Documents/Rishi/glucoscan_streamlit/images',
    skip_existing_csv='output_with_glucose.csv'
)

if len(df_new) > 0:
    # Filter out failed results (where glucose_reading starts with "Error")
    df_successful = df_new[~df_new['glucose_reading'].str.startswith('Error', na=False)].copy()
    df_failed = df_new[df_new['glucose_reading'].str.startswith('Error', na=False)].copy()
    
    if len(df_failed) > 0:
        print(f"\n⚠️  Failed to process {len(df_failed)} images:")
        print(df_failed[['filename', 'glucose_reading']])
    
    if len(df_successful) > 0:
        print(f"\n✅ Successfully processed {len(df_successful)} new images:")
        print(df_successful)
        
        # Append only successful results to the existing CSV
        existing_df = pd.read_csv('output_with_glucose.csv')
        combined_df = pd.concat([existing_df, df_successful], ignore_index=True)
        combined_df.to_csv('output_with_glucose.csv', index=False)
        print(f"\nUpdated CSV with {len(df_successful)} new images. Total records: {len(combined_df)}")
    else:
        print("\nNo successful images to add to CSV.")
else:
    print("No new images to process.")