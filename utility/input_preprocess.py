import re
import os

from unstructured.partition.pdf import partition_pdf

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.wav', '.mp3']

def extract_image_pdf(pdf_path: str) -> str:
    out_dir = f"data/images/{pdf_path.split('/')[-1].strip('.pdf')}"
    elements = partition_pdf(
                filename=pdf_path,                  
                strategy="hi_res",                                     
                extract_images_in_pdf=True,                            
                extract_image_block_types=["Image", "Table"],          
                extract_image_block_to_payload=False,                  
                extract_image_block_output_dir=out_dir,  
                )
    
    return "\n".join([str(el) for el in elements][:20])
    
def get_file_type(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in IMAGE_EXTENSIONS:
        return 'img'
    elif ext in VIDEO_EXTENSIONS:
        return 'audio'
    elif ext == '.pdf':
        return 'pdf'
    else:
        return None

def extract_img_link(prompt: str) -> str:
    url_pattern = r'(https?://(?:www\.)?\S+)'

    # Search for the pattern in the prompt
    match = re.search(url_pattern, prompt)

    # If a match is found, return the URL
    if match:
        return match.group(0)
    else:
        return None
    
def extract_youtube_link(prompt: str) -> str:
    # Define a regular expression pattern for YouTube URLs
    youtube_url_pattern = r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)'

    # Search for the pattern in the prompt
    match = re.search(youtube_url_pattern, prompt)

    # If a match is found, return the URL
    if match:
        return match.group(0)
    else:
        return None
