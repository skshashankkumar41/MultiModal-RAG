import os 
import fitz
from PIL import Image
import logging
import shutil
from src.utils.table_transformer import detect_and_crop_save_table

logger = logging.getLogger(__name__)

class PdfHandler():
    """
    A class to handle PDF processing tasks such as listing PDF files, converting PDF pages to images, 
    and extracting tables from images.

    Attributes:
        base_path (str): The base path where the PDF and image directories are located.
    """
    def __init__(self, base_path='./data'):
        self.base_path = base_path
     
    def _list_pdf_files(self):
        """
        List all PDF files in the specified directory.

        Returns:
            list: List of paths to PDF files.
        """
        pdf_files = []
        pdf_folder_path = os.path.join(self.base_path, 'pdf')
        for root, dirs, files in os.walk(pdf_folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    
    def _get_pdf_filename(self, pdf_path):
        """
        Get the filename of the PDF from its path.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The filename of the PDF.
        """
        return os.path.basename(pdf_path)
    
    def _delete_folders(self):
        """
        Delete specified folders if they exist.
        """
        try:
            paths = ['./data/images', './data/table_images']
            for path in paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Directory '{path}' successfully removed.")
        except OSError as e:
            # Handle potential errors
            print(f"Error: {path} : {e.strerror}")
        
    def _convert_pdf_to_image(self):
        """
        Convert pages of each PDF file to images and save them.
        """       
        print("Converting PDF pages to images")
        for pdf_path in self._list_pdf_files():
            pdf_document = fitz.open(pdf_path)


            for page_number in range(pdf_document.page_count):
                # Get the page
                page = pdf_document[page_number]

                # Convert the page to an image
                pix = page.get_pixmap()

                # Create a Pillow Image object from the pixmap
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                image_output_folder = f"{self.base_path}/images/"
                
                if not os.path.exists(image_output_folder):
                    os.makedirs(image_output_folder)

                # Save the image
                image.save(f"{image_output_folder}/{self._get_pdf_filename(pdf_path)}_page_{page_number + 1}.png")

            # Close the PDF file
            pdf_document.close()
            
    def _extract_tables_from_images(self):
        """
        Extract tables from images and save the cropped tables.
        """
        print("Extracting tables from images")
        for file_name in os.listdir(os.path.join(self.base_path,'images')):
            if 'png' in file_name:
                file_path = os.path.join(self.base_path, 'images' ,file_name)
                detect_and_crop_save_table(file_path, cropped_table_directory=os.path.join(self.base_path,'table_images/'))
                
    def process_pdfs(self):
        """
        Process PDFs by deleting existing folders, converting PDF pages to images, 
        and extracting tables from images.
        """
        print("Processing PDFs")
        self._delete_folders()
        self._convert_pdf_to_image()
        self._extract_tables_from_images()
        
        

            
