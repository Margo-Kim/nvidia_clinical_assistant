"""
Embeddings module with hardcoded NVIDIA configuration
"""

import logging
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

logger = logging.getLogger(__name__)

# Hardcoded configuration
EMBEDDING_URL = "http://localhost:8089"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
BATCH_SIZE = 3

def create_embeddings():
    """
    Create NVIDIA embeddings model with hardcoded configuration.
    
    Returns:
        Configured NVIDIAEmbeddings instance
    """
    logger.info(f"Setting up NVIDIA embedding model {EMBEDDING_MODEL}")
    
    try:
        embedding_model = NVIDIAEmbeddings(
            model_name=EMBEDDING_MODEL,
            base_url=EMBEDDING_URL,
            max_batch_size=BATCH_SIZE
        )
        
        logger.info("Embedding model initialized successfully")
        return embedding_model
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise
        
def test_embedding(embedding_model):
    """
    Test the embedding model with a simple text.
    
    Args:
        embedding_model: Embedding model to test
        
    Returns:
        Embedding vector or None if there was an error
    """
    logger.info("Testing embedding service")
    
    try:
        # Try to embed a simple text
        test_text = "This is a test of the NVIDIA embedding service."
        embedding = embedding_model.embed_query(test_text)
        logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error testing embedding service: {e}")
        return None