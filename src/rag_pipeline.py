from typing import List, Dict, Any
from langchain.schema import BaseRetriever, Document
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from metrics_logger import measure_inference_time, log_metrics
import re

GENERATIVE_MODEL_NAME = 'google/flan-t5-large'
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for the model

tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=GENERATIVE_MODEL_NAME,
    max_length=MAX_SEQUENCE_LENGTH,
    truncation=True
)

# Initialize text splitter with semantic chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.
    """
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def identify_figure_content(docs: List[Document], figure_number: int = None) -> List[str]:
    """
    Identify content related to specific figures in the documents.
    
    Args:
        docs: List of documents to search
        figure_number: Optional specific figure number to find
        
    Returns:
        List of text chunks related to the specified figure
    """
    figure_chunks = []
    figure_pattern = r'figure\s*(\d+)' if figure_number is None else f'figure\s*{figure_number}\\b'
    
    for doc in docs:
        content = doc.page_content
        
        # Look for figure references
        matches = re.finditer(figure_pattern, content.lower())
        for match in matches:
            # Extract the paragraph containing the figure reference
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 200)
            figure_chunk = content[start:end]
            figure_chunks.append(figure_chunk)
    
    return figure_chunks

def process_chunks(prompt_base: str, chunks: List[str], query: str) -> str:
    """
    Process text chunks and merge results intelligently.
    """
    results = []
    for chunk in chunks:
        # Clean the chunk text
        clean_chunk = clean_text(chunk)
        prompt = f"{prompt_base}{clean_chunk}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            result, _ = measure_inference_time(
                generator, 
                prompt, 
                max_length=MAX_SEQUENCE_LENGTH,
                num_return_sequences=1,
                truncation=True
            )
            if result and result[0]['generated_text']:
                results.append(result[0]['generated_text'])
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue
    
    if not results:
        return "I don't have enough information to answer that."
    
    # Merge results by removing duplicates and maintaining coherence
    merged_results = []
    seen_phrases = set()
    
    for result in results:
        # Clean and split into sentences
        clean_result = clean_text(result)
        sentences = [s.strip() for s in clean_result.split('.') if s.strip()]
        
        for sentence in sentences:
            # Create a normalized version for comparison
            normalized = ' '.join(sorted(sentence.lower().split()))
            if normalized not in seen_phrases and len(sentence) > 10:  # Filter out very short sentences
                seen_phrases.add(normalized)
                merged_results.append(sentence)
    
    if not merged_results:
        return "I don't have enough information to answer that."
    
    # Format the final response
    response = '. '.join(merged_results) + '.'
    return response

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

def is_figure_query(query: str) -> tuple:
    """
    Check if the query is asking about a specific figure.
    
    Returns:
        Tuple (is_figure_query, figure_number)
    """
    figure_match = re.search(r'figure\s*(\d+)', query.lower())
    if figure_match:
        return True, int(figure_match.group(1))
    
    if 'figure' in query.lower() or 'graph' in query.lower() or 'chart' in query.lower() or 'diagram' in query.lower() or 'table' in query.lower():
        return True, None
        
    return False, None

def run_generative_rag(query: str, retriever: BaseRetriever) -> str:
    """
    Run a generative RAG pipeline that produces an answer using a language model.
    """
    # Check if query is about figures
    is_figure, figure_number = is_figure_query(query)
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # For figure queries, use specialized processing
    if is_figure:
        figure_chunks = identify_figure_content(docs, figure_number)
        
        if figure_chunks:
            # Enhanced prompt for figure understanding
            prompt_base = """You are an expert at interpreting figures, charts, and visual data. 
            
            I'll provide you with text that describes a figure or chart from a document. 
            Based ONLY on this description and without making assumptions:
            
            1. First identify what type of figure it is (graph, chart, diagram, etc.)
            2. Describe what the figure is showing or measuring
            3. Explain the key data points or relationships shown
            4. Note any trends, patterns, or important findings highlighted
            
            If the description is incomplete or lacks specific details, acknowledge 
            the limitations instead of guessing. Only include information that is 
            explicitly stated or can be directly inferred from the text.
            
            Figure description:
            """
            
            response = process_chunks(prompt_base, figure_chunks, query)
            log_metrics(GENERATIVE_MODEL_NAME, query, response, 0)
            return response
    
    # Enhanced processing for standard queries
    prompt_base = """You are a precise and helpful research assistant. Your task is to answer 
    questions based ONLY on the provided context. Follow these steps carefully:
    
    1. First, carefully analyze the context to find relevant information
    2. Reason step-by-step to formulate your answer
    3. If the context contains sufficient information, provide a complete and accurate answer
    4. If the context lacks necessary information, clearly state what is missing
    5. Never invent or assume facts not present in the context
    
    Important: Base your answer SOLELY on the information provided. If you're uncertain, 
    indicate this rather than guessing. Accuracy is more important than completeness.
    
    Context:
    """
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Clean and split context into chunks
    clean_context = clean_text(context)
    chunks = text_splitter.split_text(clean_context)
    
    # Process chunks and merge results
    response = process_chunks(prompt_base, chunks, query)
    
    # Log metrics
    log_metrics(GENERATIVE_MODEL_NAME, query, response, 0)
    
    return response

def analyze_with_reasoning(context: str, query: str) -> str:
    """First analyze the context and apply reasoning before answering."""
    reasoning_prompt = f"""You are an analytical expert tasked with carefully examining information.
    
    Given this context:
    {context}
    
    And this question: {query}
    
    Follow this analytical process:
    1. Identify the key facts in the context relevant to the question
    2. Note any gaps or limitations in the available information
    3. Apply logical reasoning to connect the facts to the question
    4. Consider alternative interpretations if the information is ambiguous
    5. Reach a conclusion based solely on the given context
    
    Format your analysis as:
    RELEVANT FACTS: [List the key information points]
    GAPS: [Note missing information, if any]
    REASONING: [Show your step-by-step thought process]
    CONCLUSION: [State your reasoned answer]
    """
    
    # Generate reasoning
    reasoning_result = generator(reasoning_prompt, max_length=MAX_SEQUENCE_LENGTH)[0]['generated_text']
    
    # Use reasoning to generate final answer
    answer_prompt = f"""Based on this detailed analysis:
    {reasoning_result}
    
    Now provide a concise, accurate, and complete answer to the question: {query}
    
    Your answer should:
    - Be directly responsive to the question
    - Include only factual information from the analysis
    - Be clear and straightforward
    - Acknowledge uncertainty where it exists
    - Not introduce new information beyond what was in the analysis
    """
    
    final_answer = generator(answer_prompt, max_length=MAX_SEQUENCE_LENGTH)[0]['generated_text']
    return final_answer 