from typing import List
from langchain.embeddings import HuggingFaceEmbeddings

def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the sentence-transformers embedding model.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def generate_embeddings(texts: List[str], embeddings_model: HuggingFaceEmbeddings) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts (List[str]): List of texts to embed
        embeddings_model (HuggingFaceEmbeddings): Initialized embedding model
        
    Returns:
        List[List[float]]: List of embeddings
    """
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings 