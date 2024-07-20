import easyocr
from paddleocr import PaddleOCR
from ppocr.utils.logging import get_logger
from src.utils.table_transformer import recognize_table, apply_ocr, get_cell_coordinates_by_row
from PIL import Image

import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

def extract_text_from_image_v1(image_path):
    # Initialize the reader with the desired languages
    reader = easyocr.Reader(['en'])  # You can add more languages like ['en', 'es', 'fr']
    
    # Perform OCR on the image
    result = reader.readtext(image_path)
    
    # Extract text from the result
    text = ' '.join([item[1] for item in result])
    
    return text

def extract_text_from_image_v2(image_path):
    # Initialize PaddleOCR model
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the image
    result = ocr.ocr(image_path, cls=True)

    # Create and save formatted text output
    text_output = ""
    for line in result:
        line_text = " | ".join([word_info[1][0] for word_info in line])
        text_output += line_text + "\n"
        
    return text_output

def extract_text_from_image_v3(image_path):  
    try:
        cropped_table = Image.open(image_path)
        image, cells = recognize_table(cropped_table)

        cell_coordinates = get_cell_coordinates_by_row(cells)

        text = apply_ocr(cell_coordinates, image)

        
        return text
    except:
        return ''
