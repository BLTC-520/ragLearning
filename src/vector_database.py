from typing import List, Union
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever

PERSIST_DIRECTORY = './chroma_db_local'

def create_faiss_vector_database(texts: List[str], embeddings: Embeddings) -> FAISS:
    """
    Create a FAISS vector store from document chunks.
    
    Args:
        texts (List[str]): List of text chunks
        embeddings (Embeddings): Embedding model
        
    Returns:
        FAISS: FAISS vector store
    """
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def get_faiss_retriever(vectorstore: FAISS) -> BaseRetriever:
    """
    Get a retriever from a FAISS vector store.
    
    Args:
        vectorstore (FAISS): FAISS vector store
        
    Returns:
        BaseRetriever: Retriever object
    """
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def create_chroma_vector_database(texts: List[str], embeddings: Embeddings) -> Chroma:
    """
    Create and persist a Chroma vector store from document chunks.
    
    Args:
        texts (List[str]): List of text chunks
        embeddings (Embeddings): Embedding model
        
    Returns:
        Chroma: Chroma vector store
    """
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()
    return vectorstore

def load_chroma_vector_database(embeddings: Embeddings) -> Chroma:
    """
    Load a persisted Chroma vector store.
    
    Args:
        embeddings (Embeddings): Embedding model
        
    Returns:
        Chroma: Loaded Chroma vector store
    """
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

def get_chroma_retriever(vectorstore: Chroma) -> BaseRetriever:
    """
    Get a retriever from a Chroma vector store.
    
    Args:
        vectorstore (Chroma): Chroma vector store
        
    Returns:
        BaseRetriever: Retriever object
    """
    return vectorstore.as_retriever(search_kwargs={'k': 3}) 