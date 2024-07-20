import easyocr
from paddleocr import PaddleOCR
# from ppocr.utils.logging import get_logger
# import logging
# logger = get_logger()
# logger.setLevel(logging.INFO)

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
