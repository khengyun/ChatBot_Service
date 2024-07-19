import httpx
import base64
import requests
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image


def read_img(path: str):
    print(path)
    if "https" in path:
        pil_image = Image.open(BytesIO(httpx.get(path).content))
        return pil_image
    
    pil_image = Image.open(path)
    return pil_image

def array2pil(array_img):
    pil_image = Image.fromarray(array_img.astype('uint8'), 'RGB')

    return pil_image

def convert_to_base64(path: str):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    if "https" in path:
        img_str = base64.b64encode(httpx.get(path).content).decode("utf-8")
        return img_str
    
    pil_image = Image.open(path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

def main():
    file_path = "data/images/R.jpeg"
    pil_image = Image.open(file_path)

    image_b64 = convert_to_base64(pil_image)
    plt_img_base64(image_b64)

if __name__ == '__main__':
    main()