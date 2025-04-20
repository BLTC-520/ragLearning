print("Script starting...")
import argparse
from pathlib import Path
import sys

print("Imports completed")

from data_processing import load_and_split_documents, load_and_split_rich_documents
from embedding_utils import initialize_embeddings
from vector_database import (
    create_faiss_vector_database,
    get_faiss_retriever,
    create_chroma_vector_database,
    get_chroma_retriever,
    load_chroma_vector_database
)
from rag_pipeline import run_basic_rag, run_generative_rag

print("All imports successful")

def setup_rag_pipeline(data_dir: str, use_chroma: bool = False):
    """
    Set up the RAG pipeline components.
    
    Args:
        data_dir (str): Directory containing documents
        use_chroma (bool): Whether to use Chroma instead of FAISS
        
    Returns:
        tuple: (embeddings, retriever)
    """
    print(f"Starting setup with data directory: {data_dir}")
    
    # Load and split documents
    print("Loading and splitting documents...")
    text_chunks = load_and_split_documents(data_dir)
    print(f"Found {len(text_chunks)} text chunks")
    
    rich_chunks = load_and_split_rich_documents(data_dir)
    print(f"Found {len(rich_chunks)} rich document chunks")
    
    all_chunks = text_chunks + rich_chunks
    print(f"Total chunks: {len(all_chunks)}")
    
    if not all_chunks:
        print(f"No documents found in {data_dir}")
        return None, None
    
    # Initialize embeddings
    print("Initializing embedding model...")
    embeddings = initialize_embeddings()
    print("Embedding model initialized")
    
    # Create vector store
    print("Creating vector store...")
    if use_chroma:
        vectorstore = create_chroma_vector_database(all_chunks, embeddings)
        retriever = get_chroma_retriever(vectorstore)
    else:
        vectorstore = create_faiss_vector_database(all_chunks, embeddings)
        retriever = get_faiss_retriever(vectorstore)
    print("Vector store created successfully")
    
    return embeddings, retriever

def main():
    print("Entering main function")
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory containing documents")
    parser.add_argument("--use_chroma", action="store_true",
                      help="Use Chroma instead of FAISS")
    parser.add_argument("--generative", action="store_true",
                      help="Use generative RAG pipeline")
    args = parser.parse_args()
    
    print("Starting RAG pipeline...")
    
    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True)
    
    # Setup pipeline
    print("Setting up pipeline components...")
    
    # to make Freeze vector DB after creation - avoiding re-computation
    if args.use_chroma and Path('./chroma_db_local').exists():
        print("Loading cached Chroma vector DB...")
        embeddings = initialize_embeddings()
        vectorstore = load_chroma_vector_database(embeddings)
        retriever = get_chroma_retriever(vectorstore)
    else:
        embeddings, retriever = setup_rag_pipeline(str(data_dir), args.use_chroma)

    if not retriever:
        print("Failed to setup pipeline. Exiting.")
        return
    
    # Interactive loop
    print("\nRAG system is ready!")
    print("Enter your questions (type 'exit' to quit):")
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if query.lower() == 'exit':
                print("Exiting...")
                break
            
            if args.generative:
                print("Generating answer...")
                answer = run_generative_rag(query, retriever)
                print("\nGenerated Answer:")
                print(answer)
            else:
                print("Retrieving relevant documents...")
                result = run_basic_rag(query, retriever)
                print("\nRetrieved Context:")
                for i, ctx in enumerate(result['context'], 1):
                    print(f"\n--- Document {i} ---")
                    print(ctx)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    print("Starting script execution")
    main() 