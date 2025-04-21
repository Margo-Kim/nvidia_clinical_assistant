"""
Main RAG implementation that integrates all modules
"""

import logging
from typing import List, Dict, Any

from langchain.schema import Document

# Import functions from our modules
from embeddings import create_embeddings, test_embedding
from llm import create_llm
from vectorstore import create_vectorstore, search_documents
from rag_chains import create_rag_chain, query_rag_chain

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, milvus_db_path="./milvus_lite.db"):
        """
        Initialize the RAG system with hardcoded NVIDIA configurations.
        
        Args:
            milvus_db_path: Path to the Milvus Lite database file
        """
        self.milvus_db_path = milvus_db_path
        
        # Initialize components
        self.embedding_model = None
        self.chat_model = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        logger.info(f"RAG system initialized with Milvus Lite at {self.milvus_db_path}")
    
    def setup_embedding_model(self):
        """Set up NVIDIA embeddings model."""
        self.embedding_model = create_embeddings()
        return self.embedding_model
    
    def setup_chat_model(self):
        """Set up NVIDIA chat completion model."""
        self.chat_model = create_llm()
        return self.chat_model
    
    def setup_vectorstore(self, documents: List[Document]):
        """
        Set up the Milvus Lite vectorstore.
        
        Args:
            documents: List of documents to add to the vectorstore
        """
        if self.embedding_model is None:
            self.setup_embedding_model()
            
        self.vectorstore = create_vectorstore(
            documents=documents,
            embedding_model=self.embedding_model,
            milvus_db_path=self.milvus_db_path
        )
        
        # Set up retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        return self.vectorstore
    
    def setup_rag_chain(self):
        """
        Set up the RAG chain combining the retriever and LLM.
        
        Returns:
            Configured RAG chain
        """
        if self.chat_model is None:
            self.setup_chat_model()
            
        if self.retriever is None:
            logger.error("Retriever not initialized. Please set up the vector store first.")
            raise ValueError("Retriever not initialized")
        
        self.rag_chain = create_rag_chain(
            chat_model=self.chat_model,
            retriever=self.retriever
        )
        
        return self.rag_chain
    
    def search_documents(self, query_text, top_k=3):
        """
        Search for documents using the query text.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results with document text
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
            
        return search_documents(
            vectorstore=self.vectorstore,
            query_text=query_text,
            top_k=top_k
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User question
            
        Returns:
            RAG response dict containing answer and retrieved documents
        """
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("RAG chain not initialized. Please set up the RAG chain first.")
        
        return query_rag_chain(
            rag_chain=self.rag_chain,
            question=question
        )
    
    def initialize_for_documents(self, documents: List[Document]):
        """
        Initialize the complete pipeline for the provided documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Self for method chaining
        """
        logger.info("Initializing RAG pipeline...")
        
        # Setup embedding model
        self.setup_embedding_model()
        
        # Setup chat model
        self.setup_chat_model()
        
        # Setup vector store
        self.setup_vectorstore(documents)
        
        # Setup RAG chain
        self.setup_rag_chain()
        
        logger.info("RAG pipeline fully initialized")
        return self
    
    def test_embedding_service(self):
        """
        Test the NVIDIA embedding service with a simple text.
        
        Returns:
            Embedding vector or None if there was an error
        """
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        return test_embedding(self.embedding_model)
