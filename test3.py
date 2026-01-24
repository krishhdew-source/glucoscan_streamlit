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
import json
import shutil


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


def extract_glucose_batch(image_paths, batch_size=5):
    """
    Process multiple images in batches to reduce API calls and optimize performance.
    
    Args:
        image_paths: List of image file paths to process
        batch_size: Number of images to include in each API call (default: 5)
        
    Returns:
        dict: Mapping of image_path to glucose reading
    """
    results = {}
    
    if not image_paths:
        return results
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        try:
            # Build content with multiple images for single API call
            content = [
                f"Extract ONLY the primary glucose reading (big number) from each image.\n"
                f"Return a JSON array with {len(batch)} numbers in the same order as the images.\n"
                f"Example format: [115, 92, 108]\n"
                f"Return ONLY the JSON array, nothing else."
            ]
            
            # Add all images in batch to single request
            images_loaded = []
            for img_path in batch:
                try:
                    img = PIL.Image.open(img_path)
                    img = ImageOps.exif_transpose(img)
                    content.append(img)
                    images_loaded.append(img_path)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    results[img_path] = f'Error: Could not load image'
            
            if not images_loaded:
                continue
            
            # Single API call for entire batch
            print(f"Processing batch of {len(images_loaded)} images...")
            response = model.generate_content(content)
            
            try:
                # Parse JSON response
                batch_results = json.loads(response.text.strip())
                
                # Map results back to image paths
                for img_path, glucose_reading in zip(images_loaded, batch_results):
                    results[img_path] = str(glucose_reading)
                
                print(f"  ✓ Batch complete: {batch_results}")
                    
            except json.JSONDecodeError:
                # If response isn't valid JSON, try to extract numbers
                print(f"  Warning: Could not parse JSON response: {response.text}")
                for img_path in images_loaded:
                    results[img_path] = f'Error: Invalid response format'
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
            for img_path in batch:
                results[img_path] = f'Error: {str(e)}'
    
    return results


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

def extract_image_metadata_with_glucose(path, skip_existing_csv=None, batch_size=5):
    """
    Extracts image metadata (date, time) and glucose reading for all images in a directory or a single image.
    Uses batch API calls to optimize glucose extraction.
    For HEIC images, converts them to JPG first to extract metadata reliably.
    
    Args:
        path: Either a directory path or a single image file path
        skip_existing_csv: Optional path to CSV file with already processed images to skip
        batch_size: Number of images to process in each API call (default: 5)
        
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
    
    def process_single_image_metadata(image_path):
        """Helper function to extract metadata from a single image (without glucose)"""
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
                        'glucose_reading': 'Error: Could not convert HEIC',
                        'image_path': image_path
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
            
            return {
                'filename': filename,
                'date': date_str,
                'time': time_str,
                'glucose_reading': None,  # Will be filled by batch processing
                'image_path': image_path
            }
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                'filename': filename,
                'date': None,
                'time': None,
                'glucose_reading': f'Error: {str(e)}',
                'image_path': image_path
            }
        
        finally:
            # Clean up temporary JPG file
            if temp_jpg_path and os.path.exists(temp_jpg_path):
                try:
                    os.remove(temp_jpg_path)
                except:
                    pass
    
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
        # Phase 1: Collect all new images and their metadata
        print(f"Scanning directory for images...")
        metadata_results = []
        image_paths_to_process = []
        
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            
            # Skip if already in existing CSV
            if filename in existing_filenames:
                print(f"  Skipping {filename} (already processed)")
                continue
            
            # Check if it's a file
            if os.path.isfile(file_path):
                try:
                    # Try to open as an image
                    Image.open(file_path)
                    metadata_result = process_single_image_metadata(file_path)
                    metadata_results.append(metadata_result)
                    image_paths_to_process.append(file_path)
                    print(f"  Found: {filename}")
                except (IOError, OSError):
                    # Not an image file, skip it
                    pass
        
        if not image_paths_to_process:
            print("No new images to process.")
            return pd.DataFrame()
        
        print(f"\nPhase 2: Batch processing {len(image_paths_to_process)} images...")
        # Phase 2: Batch process all glucose readings in one API call (or fewer calls)
        glucose_results = extract_glucose_batch(image_paths_to_process, batch_size=batch_size)
        
        # Combine metadata with glucose readings
        for metadata in metadata_results:
            img_path = metadata['image_path']
            metadata['glucose_reading'] = glucose_results.get(img_path, 'Error: Not processed')
            del metadata['image_path']  # Remove temporary field
        
        # Return as DataFrame
        df = pd.DataFrame(metadata_results)
        return df
    
    # If it's a single file, use the original metadata extraction
    elif os.path.isfile(path):
        filename = os.path.basename(path)
        if filename in existing_filenames:
            print(f"Skipping {filename} (already processed)")
            return pd.DataFrame()
        
        metadata = process_single_image_metadata(path)
        img_path = metadata.pop('image_path')
        # For single image, use original single-image API call
        glucose_reading = extract_glucose_from_single_image(path)
        metadata['glucose_reading'] = glucose_reading
        df = pd.DataFrame([metadata])
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

csv_file_to_read = 'output_with_glucose copy.csv'

# Example usage with directory - skip already processed images
print("\nProcessing only new images (not in output_with_glucose copy.csv):")
df_new = extract_image_metadata_with_glucose(
    '/Users/richaanuragini/Documents/Rishi/glucoscan_streamlit/images',
    skip_existing_csv=csv_file_to_read
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
        existing_df = pd.read_csv(csv_file_to_read)
        combined_df = pd.concat([existing_df, df_successful], ignore_index=True)
        combined_df.to_csv(csv_file_to_read, index=False)
        print(f"\nUpdated CSV with {len(df_successful)} new images. Total records: {len(combined_df)}")
    else:
        print("\nNo successful images to add to CSV.")
else:
    print("No new images to process.")


def categorize_glucose_readings(csv_file_path):
    """
    Categorize glucose readings based on time of measurement.
    
    Args:
        csv_file_path: Path to the CSV file containing glucose readings with 'time' column
        
    Returns:
        pandas DataFrame: Original data with added 'category' column
        
    Categorization:
        - "Fasting": Readings taken before 9:30 AM
        - "Post-Breakfast": Readings from 9:30 AM to 1:00 PM
        - "Pre-Dinner": Readings from 7:00 PM to 9:00 PM
        - "Post-Dinner": Readings taken after 9:00 PM
        - "Random": Readings between 1:00 PM and 7:00 PM
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        if 'time' not in df.columns:
            print(f"Warning: 'time' column not found in {csv_file_path}")
            return df
        
        def categorize_time(time_str):
            """Helper function to categorize reading based on time"""
            if pd.isna(time_str):
                return "Uncategorized"
            
            try:
                # Parse time string (format: HH:MM:SS)
                time_obj = datetime.strptime(str(time_str).strip(), "%H:%M:%S").time()
                
                # Define time cutoffs for categorization
                morning_fasting_end = datetime.strptime("09:30:00", "%H:%M:%S").time()
                post_breakfast_end = datetime.strptime("13:00:00", "%H:%M:%S").time()
                pre_dinner_start = datetime.strptime("19:00:00", "%H:%M:%S").time()
                post_dinner_start = datetime.strptime("21:00:00", "%H:%M:%S").time()

                if time_obj < morning_fasting_end:
                    return "Fasting"
                elif time_obj < post_breakfast_end:
                    return "Post-Breakfast"
                elif time_obj >= pre_dinner_start and time_obj < post_dinner_start:
                    return "Pre-Dinner"
                elif time_obj >= post_dinner_start:
                    return "Post-Dinner"
                else:
                    return "Random" # Covers afternoon readings between breakfast and dinner
            except Exception as e:
                print(f"Warning: Could not parse time '{time_str}': {e}")
                return "Uncategorized"
        
        # Apply categorization
        df['category'] = df['time'].apply(categorize_time)
        
        print(f"\n✅ Categorized readings:")
        categories = ["Fasting", "Post-Breakfast", "Pre-Dinner", "Post-Dinner", "Random", "Uncategorized"]
        for category in categories:
            count = len(df[df['category'] == category])
            if count > 0:
                print(f"  {category}: {count}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None


# Example usage for categorizing readings
csv_file = csv_file_to_read
if os.path.exists(csv_file):
    # Create a backup copy of the original file
    backup_file = csv_file.replace('.csv', '_backup.csv')
    shutil.copy(csv_file, backup_file)
    print(f"\nCreated a backup of the original file at: {backup_file}")

    df_categorized = categorize_glucose_readings(csv_file)
    if df_categorized is not None:
        print("\nCategorized Data (last 5 rows):")
        print(df_categorized[['filename', 'time', 'glucose_reading', 'category']].tail())
        
        # Save the updated data back to the original file
        df_categorized.to_csv(csv_file, index=False)
        print(f"\n✅ Categorized data saved to {csv_file}")