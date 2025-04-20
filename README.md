# Low-Cost RAG Demo

This project demonstrates a low-cost implementation of Retrieval Augmented Generation (RAG) using open-source tools and models. It supports both basic retrieval and generative question answering over your document collection.

## Project Structure

```
.
├── data/                  # Directory for your documents
├── src/
│   ├── data_processing.py    # Document loading and chunking
│   ├── embedding_utils.py    # Embedding model initialization
│   ├── vector_database.py    # Vector store (FAISS/Chroma) operations
│   ├── rag_pipeline.py       # RAG implementation
│   └── main.py              # Main script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- Document loading support for:
  - Text files (.txt)
  - PDF files (.pdf)
  - Word documents (.docx)
- Flexible document chunking with configurable sizes and overlap
- Choice of vector stores:
  - FAISS (in-memory)
  - Chroma (persistent)
- Two RAG modes:
  - Basic retrieval (returns relevant context)
  - Generative (uses Flan-T5 to generate answers)

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your documents:
   - Place your .txt, .pdf, and .docx files in the `data/` directory
   - The system will automatically process all supported files

## Usage

### Basic Retrieval Mode

Run the system in basic retrieval mode (default):

```bash
python src/main.py
```

This will:
- Load and process documents from the `data/` directory
- Use FAISS as the vector store
- Return relevant document chunks for each query

### Generative Mode

Run the system with generative capabilities:

```bash
python src/main.py --generative
```

This will:
- Use the same document processing pipeline
- Generate natural language answers using Flan-T5

### Using Chroma Vector Store

To use Chroma instead of FAISS (for persistence):

```bash
python src/main.py --use_chroma
```

Add `--generative` for generative mode with Chroma:

```bash
python src/main.py --use_chroma --generative
```

### Custom Data Directory

Specify a different directory for your documents:

```bash
python src/main.py --data_dir /path/to/documents
```

## Further Exploration

1. Model Customization:
   - Try different embedding models by modifying `GENERATIVE_MODEL_NAME` in `rag_pipeline.py`
   - Experiment with chunk sizes and overlap in `data_processing.py`

2. Vector Store Options:
   - FAISS: Fast, in-memory vector store
   - Chroma: Persistent vector store with additional features

3. Performance Optimization:
   - Adjust retrieval parameters (k) in `vector_database.py`
   - Experiment with different chunking strategies

4. Integration Ideas:
   - Add support for more document types
   - Implement web API endpoints
   - Add document metadata tracking
   - Implement result ranking and filtering

## Contributing

Feel free to submit issues and enhancement requests! 