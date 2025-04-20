from typing import List, Dict, Any
from langchain.schema import BaseRetriever, Document
from transformers import pipeline, AutoTokenizer
from metrics_logger import measure_inference_time, log_metrics

GENERATIVE_MODEL_NAME = 'google/flan-t5-small'
MAX_INPUT_LENGTH = 450  # Reduced from 512 to leave room for generation

tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=GENERATIVE_MODEL_NAME,
    max_length=512,
    max_input_length=MAX_INPUT_LENGTH
)

def truncate_context_to_fit(prompt_base: str, context: str, max_tokens: int = MAX_INPUT_LENGTH) -> str:
    """
    Truncate context so the total prompt stays within model's token limit.
    """
    # First, encode the base prompt to know how many tokens we have available
    base_tokens = len(tokenizer.encode(prompt_base))
    available_tokens = max_tokens - base_tokens - 50  # Reserve 50 tokens for question and answer tags
    
    while True:
        input_ids = tokenizer.encode(context, truncation=False)
        if len(input_ids) <= available_tokens:
            return context
        # Remove 20% of the context each time for faster convergence
        context_len = len(context)
        context = context[:int(context_len * 0.8)]

def run_basic_rag(query: str, retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Run a basic RAG pipeline that returns relevant documents and the query.
    
    Args:
        query (str): User query
        retriever (BaseRetriever): Document retriever
        
    Returns:
        Dict[str, Any]: Dictionary containing query and retrieved context
    """
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Format response
    response = {
        'query': query,
        'context': [doc.page_content for doc in docs],
        'source_documents': docs
    }
    
    return response

def run_generative_rag(query: str, retriever: BaseRetriever) -> str:
    """
    Run a generative RAG pipeline that produces an answer using a language model.
    
    Args:
        query (str): User query
        retriever (BaseRetriever): Document retriever
        
    Returns:
        str: Generated answer
    """
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Format context with truncation
    prompt_base = """Using the following context, answer the question. If you cannot find 
    the answer in the context, say "I don't have enough information to answer that."

    Context:
    """
    context = "\n\n".join([doc.page_content for doc in docs])
    context = truncate_context_to_fit(prompt_base, context)
    
    prompt = f"{prompt_base}{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate answer with stricter length controls
    result, duration = measure_inference_time(generator, prompt, max_length=512, num_return_sequences=1)
    response = result[0]['generated_text']

    log_metrics(GENERATIVE_MODEL_NAME, query, response, duration)
    
    return response 