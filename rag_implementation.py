"""
RAG Implementation using NVIDIA AI Endpoints and Milvus Lite
"""

import os
import time
from typing import List, Dict, Any

# LangChain and related imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_milvus import Milvus
from pymilvus import MilvusClient

# Set logging level
import logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to see more detailed logs
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
        self.milvus_client = None
        self.collection_name = f"nvidia_collection_{int(time.time())}"
        
        # Connect to Milvus Lite
        self.milvus_client = MilvusClient(uri=self.milvus_db_path)
        logger.info(f"Connected to Milvus Lite database at {self.milvus_db_path}")
    
    def setup_embedding_model(self):
        """Set up NVIDIA embeddings model with detailed logging."""
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
    
    def _create_milvus_collection(self, dimension=1024):
        """Create a Milvus Lite collection with the specified dimension."""
        logger.info(f"Creating Milvus Lite collection '{self.collection_name}' with dimension {dimension}")
        try:
            # Create the collection
            if self.collection_name not in self.milvus_client.list_collections():
                self.milvus_client.create_collection(
                    collection_name=self.collection_name,
                    dimension=dimension
                )
                logger.info(f"Created Milvus Lite collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating Milvus collection: {e}")
            raise
    
    def embed_and_store_documents(self, documents: List[Document]):
        """
        Embed documents using NVIDIA embeddings and store them in Milvus Lite.
        
        Args:
            documents: List of Document objects
        """
        logger.info(f"Embedding and storing {len(documents)} documents")
        
        # Ensure embedding model is initialized
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        # Extract text content from documents
        texts = [doc.page_content for doc in documents]
        
        # Create the collection with appropriate dimension
        # NVIDIA nv-embedqa-e5-v5 uses 1024 dimensions
        self._create_milvus_collection(dimension=1024)
        
        # Process documents in small batches to avoid potential issues
        batch_size = 5
        
        try:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_texts = [doc.page_content for doc in batch_docs]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch_texts)} documents)")
                
                try:
                    # Generate embeddings for this batch
                    batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                    
                    # Prepare data for Milvus insertion
                    data_to_insert = []
                    for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                        entry = {
                            "id": i + j,
                            "vector": embedding,
                            "text": doc.page_content,
                            "metadata": str(doc.metadata)  # Convert metadata to string for storage
                        }
                        data_to_insert.append(entry)
                    
                    # Insert data into Milvus
                    self.milvus_client.insert(
                        collection_name=self.collection_name,
                        data=data_to_insert
                    )
                    
                    logger.info(f"Successfully inserted batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
            
            logger.info(f"Completed embedding and storing documents")
            
        except Exception as e:
            logger.error(f"Error in embed_and_store_documents: {e}")
            raise
    
    def setup_vectorstore(self, documents: List[Document]):
        """
        Set up the LangChain Milvus vectorstore.
        
        Args:
            documents: List of documents used to create the vectorstore
        """
        logger.info("Setting up LangChain vectorstore")
        
        try:
            # First, embed and store the documents directly
            self.embed_and_store_documents(documents)
            
            # Then create a LangChain Milvus vectorstore for the same collection
            if self.embedding_model is None:
                self.setup_embedding_model()
                
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={"uri": self.milvus_db_path}
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
            
            logger.info("Vectorstore and retriever setup complete")
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
        
        # Execute the chain
        response = self.rag_chain.invoke({"input": question})
        
        return response
    
    def initialize_for_documents(self, documents: List[Document]):
        """
        Initialize the complete pipeline for the provided documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Self for method chaining
        """
        logger.info("Initializing NVIDIA RAG pipeline...")
        
        # Setup embedding model
        self.setup_embedding_model()
        
        # Setup chat model
        self.setup_chat_model()
        
        # Setup vector store and insert documents
        self.setup_vectorstore(documents)
        
        # Setup RAG chain
        self.setup_rag_chain()
        
        logger.info("NVIDIA RAG pipeline fully initialized")
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

    def list_collections(self):
        """List all collections in Milvus Lite"""
        collections = self.milvus_client.list_collections()
        logger.info(f"Collections in Milvus Lite: {collections}")
        return collections
    
    def search_documents(self, query_text, top_k=3):
        """
        Search for documents using the query text
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{query_text}'")
        
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Search using the embedding
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            logger.info(f"Found {len(results[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise


# Example usage
def example():
    # Create some test documents
    documents = [
        Document(page_content="Artificial intelligence was founded as an academic discipline in 1956.", 
                 metadata={"source": "test", "index": 0}),
        Document(page_content="Alan Turing was the first person to conduct substantial research in AI.", 
                 metadata={"source": "test", "index": 1}),
        Document(page_content="Born in Maida Vale, London, Turing was raised in southern England.", 
                 metadata={"source": "test", "index": 2}),
    ]
    
    # Initialize the RAG system
    rag = NVIDIARAGSystem()
    
    # Test the embedding service
    embedding = rag.test_embedding_service()
    if embedding:
        print(f"Embedding service is working! Vector dimension: {len(embedding)}")
    else:
        print("Embedding service test failed")
        return
    
    # Initialize the pipeline
    rag.initialize_for_documents(documents)
    
    # Test a query
    query = "Who was Alan Turing?"
    results = rag.search_documents(query)
    print(f"\nSearch results for '{query}':")
    print(results)
    
    # Test the RAG chain
    response = rag.query(query)
    print("\nRAG response:")
    print(response["answer"])

if __name__ == "__main__":
    example()