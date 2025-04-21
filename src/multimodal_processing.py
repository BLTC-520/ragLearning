from typing import List, Dict, Any
from pathlib import Path
import os
from PIL import Image
import pytesseract
from langchain.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {str(e)}")
        return ""

def process_multimodal_document(file_path: str) -> Dict[str, Any]:
    """
    Process a document that may contain both text and images.
    
    Args:
        file_path (str): Path to the document
        
    Returns:
        Dict[str, Any]: Dictionary containing text and image information
    """
    result = {
        "text": "",
        "images": [],
        "metadata": {}
    }
    
    file_ext = Path(file_path).suffix.lower()
    
    # Load document based on type
    if file_ext == '.pdf':
        loader = UnstructuredPDFLoader(file_path)
    elif file_ext in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    
    # Extract text content
    documents = loader.load()
    result["text"] = "\n".join([doc.page_content for doc in documents])
    
    # Extract images if present
    if file_ext == '.pdf':
        # TODO: Implement PDF image extraction
        pass
    elif file_ext in ['.docx', '.doc']:
        # TODO: Implement DOCX image extraction
        pass
    
    # Add metadata
    result["metadata"] = {
        "file_path": file_path,
        "file_type": file_ext,
        "file_size": os.path.getsize(file_path)
    }
    
    return result

def process_multimodal_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Process all documents in a directory, handling both text and images.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        List[Dict[str, Any]]: List of processed documents
    """
    processed_docs = []
    
    # Supported file extensions
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg']
    
    # Process all files in directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in supported_extensions:
                if file_ext in ['.png', '.jpg', '.jpeg']:
                    # Process image file
                    text = extract_text_from_image(file_path)
                    processed_docs.append({
                        "text": text,
                        "images": [file_path],
                        "metadata": {
                            "file_path": file_path,
                            "file_type": file_ext,
                            "file_size": os.path.getsize(file_path)
                        }
                    })
                else:
                    # Process document file
                    processed_docs.append(process_multimodal_document(file_path))
    
    return processed_docs 