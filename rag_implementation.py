"""
RAG Implementation using NVIDIA AI Endpoints and Milvus Lite
"""

import os
import time
import logging
from typing import List, Dict, Any

# LangChain and related imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_milvus import Milvus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 milvus_db_path: str = "./milvus_lite.db",
                 embedding_base_url: str = "http://localhost:8089",
                 embedding_model_name: str = "nvidia/nv-embedqa-e5-v5",
                 llm_base_url: str = "http://localhost:8088",
                 llm_model_name: str = "meta/llama-3.1-nemotron-70b-instruct:latest"):
        """
        Initialize the RAG system with NVIDIA embeddings.
        
        Args:
            milvus_db_path: Path to the Milvus Lite database file
            embedding_base_url: NVIDIA embedding service URL
            embedding_model_name: NVIDIA embedding model name
            llm_base_url: NVIDIA LLM service URL
            llm_model_name: NVIDIA LLM model name
        """
        self.milvus_db_path = milvus_db_path
        self.embedding_base_url = embedding_base_url
        self.embedding_model_name = embedding_model_name
        self.llm_base_url = llm_base_url
        self.llm_model_name = llm_model_name
        
        # Initialize components
        self.embedding_model = None
        self.chat_model = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.collection_name = f"rag_collection_{int(time.time())}"
        
        logger.info(f"RAG system initialized with Milvus Lite at {self.milvus_db_path}")
    
    def setup_embedding_model(self):
        """Set up NVIDIA embeddings model."""
        logger.info(f"Setting up NVIDIA embedding model {self.embedding_model_name} at {self.embedding_base_url}")
        try:
            self.embedding_model = NVIDIAEmbeddings(
                model_name=self.embedding_model_name,
                base_url=self.embedding_base_url,
                max_batch_size=10  # Lower batch size to help with stability
            )
            logger.info("Embedding model initialized successfully")
            
            # Log available models for debugging
            try:
                available_models = self.embedding_model.available_models
                logger.info(f"Available embedding models: {[model.id for model in available_models]}")
            except Exception as e:
                logger.warning(f"Could not fetch available models: {e}")
                
            return self.embedding_model
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def setup_chat_model(self):
        """Set up NVIDIA chat completion model."""
        logger.info(f"Setting up NVIDIA chat model {self.llm_model_name} at {self.llm_base_url}")
        try:
            self.chat_model = ChatNVIDIA(
                model_name=self.llm_model_name,
                base_url=self.llm_base_url,
                max_tokens=32768,
                temperature=0.1,
                model_kwargs={
                    "do_sample": True,
                    "top_p": 0.9
                }
            )
            logger.info("Chat model initialized successfully")
            return self.chat_model
        except Exception as e:
            logger.error(f"Error initializing chat model: {e}")
            raise
    
    def setup_vectorstore(self, documents: List[Document]):
        """
        Set up the LangChain Milvus vectorstore.
        
        Args:
            documents: List of documents to add to the vectorstore
        """
        logger.info("Setting up LangChain Milvus vectorstore")
        
        try:
            if self.embedding_model is None:
                self.setup_embedding_model()
                
            # Use from_documents to create and populate the vector store
            self.vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_db_path},
                drop_old=True      # Create a new collection
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
            
            logger.info("LangChain Milvus vectorstore setup successful")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error setting up vectorstore: {e}")
            raise
    
    def setup_rag_chain(self):
        """
        Set up the RAG chain combining the retriever and LLM.
        
        Returns:
            Configured RAG chain
        """
        logger.info("Setting up RAG chain")
        
        # Ensure chat model is initialized
        if self.chat_model is None:
            self.setup_chat_model()
            
        # Ensure retriever is set up
        if self.retriever is None:
            logger.error("Retriever not initialized. Please set up the vector store first.")
            raise ValueError("Retriever not initialized")
        
        # Create prompt template for the QA chain
        prompt = ChatPromptTemplate.from_template("""
        <context>
        {context}
        </context>
        
        Based on the provided context, please answer the following question accurately and concisely.
        If the answer cannot be found in the context, acknowledge that and provide general information if possible.
        
        Question: {input}
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.chat_model, prompt)
        
        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(self.retriever, document_chain)
        
        logger.info("RAG chain initialized")
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
        logger.info(f"Searching for: '{query_text}'")
        
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
                
            results = self.vectorstore.similarity_search(query_text, k=top_k)
            logger.info(f"Search returned {len(results)} results")
            return {"documents": results}
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"documents": [], "error": str(e)}
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User question
            
        Returns:
            RAG response dict containing answer and retrieved documents
        """
        # Ensure RAG chain is initialized
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("RAG chain not initialized. Please set up the RAG chain first.")
        
        logger.info(f"Processing query: {question}")
        
        try:
            # Execute the chain
            response = self.rag_chain.invoke({"input": question})
            return response
        except Exception as e:
            logger.error(f"Error executing RAG chain: {e}")
            # Return a fallback response with error information
            return {
                "answer": f"I encountered an error processing your query. Error: {str(e)}",
                "context": []
            }
    
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
    
    def test_embedding_service(self, test_text="This is a test of the NVIDIA embedding service."):
        """
        Test the NVIDIA embedding service with a simple text.
        
        Args:
            test_text: Text to embed for testing
        
        Returns:
            Embedding vector or None if there was an error
        """
        logger.info("Testing NVIDIA embedding service")
        
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        try:
            # Try to embed a simple text
            embedding = self.embedding_model.embed_query(test_text)
            logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error testing embedding service: {e}")
            return None