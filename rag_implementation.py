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
from langchain.chains import LLMChain
from pymilvus import MilvusClient

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
        self.milvus_client = None
        self.collection_name = f"rag_collection_{int(time.time())}"
        
        # Try both approaches - first try to connect with MilvusClient for direct operations
        try:
            self.milvus_client = MilvusClient(uri=self.milvus_db_path)
            logger.info(f"Connected to Milvus Lite database directly at {self.milvus_db_path}")
        except Exception as e:
            logger.warning(f"Could not connect directly to Milvus Lite: {e}")
        
        # Also try to establish LangChain connection for later use
        try:
            # This is just to test the connection, we'll create the actual vectorstore later
            self.test_langchain_connection()
            logger.info("LangChain Milvus connection test successful")
        except Exception as e:
            logger.warning(f"LangChain Milvus connection test failed: {e}")
    
    def test_langchain_connection(self):
        """Test if we can connect to Milvus via LangChain."""
        try:
            # Create a temporary embedding model if needed
            if self.embedding_model is None:
                self.setup_embedding_model()
            
            # Try to create a simple Milvus instance just to test connection
            test_store = Milvus(
                embedding_function=self.embedding_model,
                connection_args={"uri": self.milvus_db_path},
                collection_name="test_connection",
                drop_old=False
            )
            # If no error was raised, connection works
            return True
        except Exception as e:
            logger.error(f"LangChain Milvus connection test failed: {e}")
            raise
    
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
    
    def _create_milvus_collection(self, dimension=1024):
        """Create a Milvus Lite collection with the specified dimension."""
        if self.milvus_client is None:
            logger.error("Direct Milvus client not available")
            return False
            
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
            return True
        except Exception as e:
            logger.error(f"Error creating Milvus collection: {e}")
            return False
    
    def setup_vectorstore_langchain(self, documents: List[Document]):
        """
        Try to set up the LangChain Milvus vectorstore with explicit parameters.
        
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
                text_field="text",  # Explicitly set text field
                drop_old=True,      # Create a new collection
                index_params={"index_type": "FLAT", "metric_type": "L2"}  # Only FLAT is supported in Milvus Lite
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
            
            logger.info("LangChain Milvus vectorstore setup successful")
            return True
        except Exception as e:
            logger.error(f"Error setting up LangChain Milvus vectorstore: {e}")
            return False
    
    def embed_and_store_documents(self, documents: List[Document]):
        """
        Embed documents using NVIDIA embeddings and store them in Milvus Lite using direct client.
        
        Args:
            documents: List of Document objects
        """
        if self.milvus_client is None:
            logger.error("Direct Milvus client not available")
            return False
            
        logger.info(f"Embedding and storing {len(documents)} documents using direct client")
        
        # Ensure embedding model is initialized
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        # Create the collection with appropriate dimension
        # NVIDIA nv-embedqa-e5-v5 uses 1024 dimensions
        self._create_milvus_collection(dimension=1024)
        
        # Process documents in small batches
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
                        # Store the actual document content directly
                        entry = {
                            "id": i + j,
                            "vector": embedding,
                            "text": doc.page_content,  
                            "source": doc.metadata.get("source", "unknown"),
                            "index": doc.metadata.get("index", 0)
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
            
            logger.info(f"Completed embedding and storing documents with direct client")
            return True
            
        except Exception as e:
            logger.error(f"Error in embed_and_store_documents: {e}")
            return False
    
    def setup_rag_chain_langchain(self):
        """
        Try to set up the RAG chain using LangChain's built-in functionality.
        
        Returns:
            Boolean indicating success or failure
        """
        logger.info("Setting up LangChain RAG chain")
        
        # Ensure chat model is initialized
        if self.chat_model is None:
            self.setup_chat_model()
            
        # Ensure retriever is set up
        if self.retriever is None:
            logger.error("Retriever not initialized. Please set up the vector store first.")
            return False
        
        try:
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
            
            logger.info("LangChain RAG chain initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up LangChain RAG chain: {e}")
            return False
    
    def create_direct_rag_chain(self):
        """
        Create a simple LLM chain for processing RAG queries directly
        
        Returns:
            Configured LLM chain
        """
        logger.info("Creating direct RAG chain")
        
        # Ensure chat model is initialized
        if self.chat_model is None:
            self.setup_chat_model()
        
        # Create a prompt template for RAG
        prompt = ChatPromptTemplate.from_template("""
        <context>
        {context}
        </context>
        
        Based on the provided context, please answer the following question accurately and concisely.
        If the answer cannot be found in the context, acknowledge that and provide general information if possible.
        
        Question: {question}
        """)
        
        # Create an LLM chain
        chain = LLMChain(
            llm=self.chat_model,
            prompt=prompt
        )
        
        logger.info("Direct RAG chain created successfully")
        return chain
    
    def search_documents(self, query_text, top_k=3):
        """
        Search for documents using the query text directly through Milvus.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of search results with document text
        """
        logger.info(f"Searching for: '{query_text}'")
        
        # Try using LangChain vectorstore first
        if self.vectorstore is not None:
            try:
                logger.info("Attempting search with LangChain vectorstore")
                results = self.vectorstore.similarity_search(query_text, k=top_k)
                logger.info(f"LangChain search returned {len(results)} results")
                return {"results": None, "documents": results, "source": "langchain"}
            except Exception as e:
                logger.warning(f"LangChain vectorstore search failed: {e}")
                # Fall back to direct search
                
        # Fall back to direct search if LangChain failed or vectorstore isn't initialized
        if self.milvus_client is not None:
            try:
                logger.info("Attempting direct search with MilvusClient")
                
                if self.embedding_model is None:
                    self.setup_embedding_model()
                
                # Generate embedding for query
                query_embedding = self.embedding_model.embed_query(query_text)
                
                # Search using the embedding
                results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=["text", "source", "index"]
                )
                
                # Create Document objects from the search results
                documents = []
                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit.get("entity", {})
                        doc = Document(
                            page_content=entity.get("text", ""),
                            metadata={
                                "source": entity.get("source", "unknown"),
                                "index": entity.get("index", 0),
                                "score": hit.get("score", 0)
                            }
                        )
                        documents.append(doc)
                
                logger.info(f"Direct search returned {len(documents)} documents")
                return {"results": results, "documents": documents, "source": "direct"}
                
            except Exception as e:
                logger.error(f"Direct search failed: {e}")
                
        # If all search methods failed
        logger.error("All search methods failed")
        return {"results": None, "documents": [], "error": "All search methods failed"}
    
    def query(self, question: str, top_k=4) -> Dict[str, Any]:
        """
        Process a question through RAG with fallback mechanisms.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            RAG response dict containing answer and retrieved documents
        """
        logger.info(f"Processing query: {question}")
        
        # Try LangChain RAG chain first
        if self.rag_chain is not None:
            try:
                logger.info("Attempting to use LangChain RAG chain")
                response = self.rag_chain.invoke({"input": question})
                logger.info("LangChain RAG chain succeeded")
                return response
            except Exception as e:
                logger.warning(f"LangChain RAG chain failed: {e}")
                # Fall back to direct approach
        
        # Fall back to direct approach
        try:
            logger.info("Falling back to direct RAG approach")
            
            # 1. Search for relevant documents
            search_result = self.search_documents(question, top_k=top_k)
            documents = search_result.get("documents", [])
            
            if not documents:
                return {
                    "answer": "I couldn't find any relevant information in the database.",
                    "context": []
                }
            
            # 2. Create the context from retrieved documents
            context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
            
            # 3. Create the chain if needed
            chain = self.create_direct_rag_chain()
            
            # 4. Generate the response
            response = chain.run({"context": context_text, "question": question})
            
            return {
                "answer": response,
                "context": documents
            }
            
        except Exception as e:
            logger.error(f"Error in direct RAG query process: {e}")
            return {
                "answer": f"I encountered an error processing your query. Error: {str(e)}",
                "context": []
            }
    
    def initialize_for_documents(self, documents: List[Document]):
        """
        Initialize the complete pipeline for the provided documents,
        trying both LangChain and direct approaches.
        
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
        
        # Try LangChain vectorstore setup
        langchain_success = self.setup_vectorstore_langchain(documents)
        
        # If LangChain setup fails or as a backup, also do direct setup
        direct_success = self.embed_and_store_documents(documents)
        
        if langchain_success:
            # Try setting up LangChain RAG chain
            self.setup_rag_chain_langchain()
        
        if langchain_success or direct_success:
            logger.info("RAG pipeline initialization complete")
            return self
        else:
            logger.error("Failed to initialize RAG pipeline through any method")
            raise RuntimeError("Failed to initialize RAG pipeline")
    
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
        if self.milvus_client:
            collections = self.milvus_client.list_collections()
            logger.info(f"Collections in Milvus Lite: {collections}")
            return collections
        else:
            logger.error("Direct Milvus client not available")
            return []