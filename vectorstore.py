"""
Vector store module for Milvus Lite
"""

import logging
import time
from typing import List

from langchain.schema import Document
from langchain_milvus import Milvus

logger = logging.getLogger(__name__)

def create_vectorstore(documents, embedding_model, milvus_db_path):
    """
    Create a Milvus Lite vector store with the provided documents.
    
    Args:
        documents: List of documents to add to the vector store
        embedding_model: Embedding model to use
        milvus_db_path: Path to the Milvus Lite database file
    
    Returns:
        Configured Milvus vector store
    """
    logger.info("Setting up Milvus Lite vector store")
    
    # Generate collection name
    collection_name = f"rag_collection_{int(time.time())}"
    
    try:
        # Create a Milvus vector store with Milvus Lite
        # Explicitly specify FLAT index type which is supported by Milvus Lite
        vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            connection_args={"uri": milvus_db_path},
            index_params={"index_type": "FLAT", "metric_type": "L2"},  # Milvus Lite only supports FLAT, IVF_FLAT, AUTOINDEX
            drop_old=True
        )
        
        logger.info(f"Created Milvus Lite vector store with collection: {collection_name}")
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error creating Milvus Lite vector store: {e}")
        raise

def search_documents(vectorstore, query_text, top_k=3):
    """
    Search for documents using the vector store.
    
    Args:
        vectorstore: The Milvus vector store
        query_text: Query text
        top_k: Number of results to return
    
    Returns:
        List of search results
    """
    logger.info(f"Searching for: '{query_text}'")
    
    try:
        results = vectorstore.similarity_search(query_text, k=top_k)
        logger.info(f"Search returned {len(results)} results")
        return {"documents": results}
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return {"documents": [], "error": str(e)}