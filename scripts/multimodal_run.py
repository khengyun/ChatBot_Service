import os
import argparse

from typing import List

from utility import extract_image_pdf
from bot import MultiModalBot

# Create CLI.
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default = None, type=str, help="Path to image/video/pdf.")
parser.add_argument("--enabled", default = None, type=str, help="Input data type.")
parser.add_argument("--prompt", default = None, type=str, help="Input prompt.")
args = parser.parse_args()
print(args.file_path, args.enabled)

def get_multimodal_response(file_path: str, prompt: str, enabled: str) -> str:
    output = None
    if enabled == "img":
        multimodal_model = MultiModalBot(path=file_path)
        output = multimodal_model.describe(prompt)    
    elif enabled == "audio":
        multimodal_model = MultiModalBot(path=file_path, whisper_model_id='base')
        if 'http' in file_path:
            multimodal_model.download_mp4_from_youtube()
        output = multimodal_model.summarize()
        print(output)
    elif enabled == "pdf":
        output = extract_image_pdf(pdf_path=file_path)

    return output

desc_summ = get_multimodal_response(file_path=args.file_path, prompt=args.prompt, enabled=args.enabled)
print(desc_summ)
with open("data/desc_summ.txt", "w") as f:
    f.write(args.enabled)
    f.write("\n")
    f.write(desc_summ)
