from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)

def load_and_split_documents(directory_path: str) -> List[str]:
    """
    Load all .txt files from a directory and split them into chunks.
    
    Args:
        directory_path (str): Path to the directory containing text files
        
    Returns:
        List[str]: List of text chunks
    """
    # Create directory loader for .txt files
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    
    # Load documents
    documents = txt_loader.load()
    
    # Create text splitter with enhanced chunking options
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Hard maximum size
        chunk_overlap=50,  # Overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Preserve natural boundaries
        keep_separator=True,  # Keep separators in chunks
        is_separator_regex=False  # Use exact string matching
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def load_and_split_rich_documents(directory_path: str) -> List[str]:
    """
    Load all .pdf and .docx files from a directory and split them into chunks.
    
    Args:
        directory_path (str): Path to the directory containing PDF and DOCX files
        
    Returns:
        List[str]: List of text chunks
    """
    # Create directory loaders for PDF and DOCX files
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    docx_loader = DirectoryLoader(
        directory_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )
    
    # Load documents
    documents = pdf_loader.load() + docx_loader.load()
    
    # Create token splitter with enhanced chunking options
    token_splitter = TokenTextSplitter(
        chunk_size=500,  # Hard maximum size
        chunk_overlap=100,  # Overlap between chunks
        encoding_name="cl100k_base",  # Tokenizer to use
        add_start_index=True  # Add start index to chunks
    )
    
    # Split documents
    chunks = token_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks] 