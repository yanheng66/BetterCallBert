from llama_index.core import Document, SimpleDirectoryReader
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def load_pdf_text(path):
    return SimpleDirectoryReader(input_files=[path]).load_data()

def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_texts = []
    for i, image in enumerate(images):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary)
        extracted_texts.append(text)
    return Document(text="\n".join(extracted_texts))
