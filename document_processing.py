"""
Document processing module for FiQA dataset
"""

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datasets import load_dataset
from tiktoken import get_encoding
# choose reasonable defaults for FiQA
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,         # ≈ 480‑token ceiling keeps NIM happy
    chunk_overlap=120,
)

ENC = get_encoding("cl100k_base")
MAX_TOKENS = 480   
logger = logging.getLogger(__name__)

def load_fiqa_dataset(limit=10):
    """
    Load the FiQA dataset using the Hugging Face datasets library
    
    Args:
        limit: Number of samples to load
        
    Returns:
        Dataset sample or None if there was an error
    """
    logger.info("Loading the FiQA dataset from Hugging Face...")
    
    try:
        # Load the FiQA corpus dataset
        dataset = load_dataset("explodinggradients/fiqa", "corpus")
        logger.info(f"Dataset structure: {dataset}")
        
        # For demonstration purposes, use a small subset
        sample_data = dataset["corpus"].select(range(limit))
        
        logger.info(f"Loaded {len(sample_data)} samples from the FiQA corpus dataset")
        return sample_data
    
    except Exception as e:
        logger.error(f"Error loading FiQA dataset: {e}")
        return None

def convert_fiqa_to_documents(fiqa_data):
    """
    Convert FiQA data to LangChain Document format
    
    Args:
        fiqa_data: FiQA dataset
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for item in fiqa_data:
        # The FiQA corpus dataset has a field called 'doc' that contains the text
        content = item["doc"]
        
        # Create metadata
        metadata = {
            "source": "fiqa_corpus",
            "index": len(documents)
        }
        
        # Create the document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    logger.info(f"Converted {len(documents)} FiQA items to Document format")
    return documents

# def split_documents(docs, splitter=_SPLITTER):
#     """
#     Break long FiQA documents into smaller overlapping chunks.

#     Args:
#         docs (list[Document]): full‑length FiQA docs
#         splitter: any LangChain TextSplitter

#     Returns:
#         list[Document]: chunked docs with updated metadata
#     """
#     chunked_docs = []

#     for doc in docs:
#         for i, chunk in enumerate(splitter.split_text(doc.page_content)):
#             chunked_docs.append(
#                 Document(
#                     page_content=chunk,
#                     metadata={**doc.metadata, "chunk": i}
#                 )
#             )
#     logger.info(f"Split {len(docs)} documents into {len(chunked_docs)} chunks")
#     # print(chunked_docs[0:30])
#     return chunked_docs
def split_documents(raw_docs):
    docs = []
    for doc in raw_docs:
        for i, chunk in enumerate(_SPLITTER.split_text(doc.page_content)):
            if len(chunk) <= 450:          # character guard
                docs.append(
                    Document(page_content=chunk,
                             metadata={**doc.metadata, "chunk": i})
                )
    logger.info("Split %d docs into %d safe chunks", len(raw_docs), len(docs))
    return docs

def process_documents(limit=10):
    """
    Process FiQA documents from loading to conversion
    
    Args:
        limit: Number of samples to load
        
    Returns:
        List of Document objects
    """
    # Load dataset
    fiqa_data = load_fiqa_dataset(limit=limit)
    
    if fiqa_data is None or len(fiqa_data) == 0:
        logger.error("Could not load FiQA dataset")
        return []
    
    # Convert to documents
    documents = convert_fiqa_to_documents(fiqa_data)
    return split_documents(documents)