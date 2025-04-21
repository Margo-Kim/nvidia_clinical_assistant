"""
Document processing module for FiQA dataset
"""

import logging
from typing import List

from langchain.schema import Document
from datasets import load_dataset

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
    return documents