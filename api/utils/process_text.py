import re , os
import pandas as pd
import PyPDF2
import re
from pptx import Presentation
import pytesseract
from pdf2image import convert_from_path
import logging
logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> str:
    """Automatically detect file type and extract cleaned text."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        text = extract_and_clean_pdf(file_path)
        if not text.strip():  # fallback to OCR
            text = extract_text_from_pdf_with_ocr(file_path)
        return text
    elif ext in [".ppt", ".pptx"]:
        return extract_and_clean_ppt(file_path)
    elif ext in [".xls", ".xlsx"]:
        return extract_and_clean_excel(file_path)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""

# -----------------------------
#  Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """Cleans extracted text by removing unwanted symbols and spaces."""
    text = re.sub(r'[^A-Za-z0-9\s.,]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -----------------------------
#  PDF Extraction
# -----------------------------
def extract_and_clean_pdf(file_path: str) -> str:
    """Extracts text from normal PDF (non-scanned)."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
        return clean_text(extracted_text)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        return ""
    

def extract_text_from_pdf_with_ocr(pdf_path: str) -> str:
    """Extracts text from scanned PDF using OCR."""
    try:
        logger.info(f"Starting OCR extraction for PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        extracted_text = []

        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with OCR")
            page_text = pytesseract.image_to_string(image)
            page_text = clean_text(page_text)
            if page_text.strip():
                extracted_text.append(page_text)

        full_text = "\n\n".join(extracted_text)
        if not full_text.strip():
            logger.warning("OCR extraction produced no text")
        return full_text
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return ""


# -----------------------------
#  PPT Extraction
# -----------------------------
def extract_and_clean_ppt(file_path: str) -> str:
    """Extracts text from PowerPoint slides."""
    try:
        presentation = Presentation(file_path)
        extracted_text = []
        for slide in presentation.slides:
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text.strip())
            extracted_text.append("\n".join(slide_text))
        return clean_text("\n\n".join(extracted_text))
    except Exception as e:
        logger.error(f"PPT extraction failed: {str(e)}")
        return ""

# -----------------------------
#  Excel Extraction
# -----------------------------
def extract_and_clean_excel(file_path: str) -> str:
    """Extracts and cleans text from Excel files."""
    try:
        df = pd.read_excel(file_path)
        df = df.dropna(how='all').fillna('')
        cleaned_text = " ".join(
            df.applymap(lambda x: clean_text(str(x))).astype(str).values.flatten()
        )
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"Excel extraction failed: {str(e)}")
        return ""
    
def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    